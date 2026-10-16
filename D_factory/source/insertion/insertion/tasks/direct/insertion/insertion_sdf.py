# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Signed-distance queries and interpenetration measurements for insertion meshes."""

from __future__ import annotations

import os

import torch

from . import insertion_math as im

_wp = None
_KERNEL = None


def _warp():
    """Import Warp, initialise it once, build the kernel once."""
    global _wp, _KERNEL
    if _wp is None:
        import warp as wp

        wp.init()

        @wp.kernel
        def _sdf_batched(
            mesh: wp.uint64,
            points: wp.array(dtype=wp.vec3),
            xf_pos: wp.array(dtype=wp.vec3),
            xf_quat: wp.array(dtype=wp.quat),
            max_dist: float,
            out: wp.array2d(dtype=wp.float32),
        ):
            """Signed distance of every sampled point, for every env."""
            e, i = wp.tid()
            xform = wp.transform(xf_pos[e], xf_quat[e])
            q = wp.transform_point(xform, points[i])

            sign = float(0.0)
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            res = wp.mesh_query_point(mesh, q, max_dist, sign, face_index, face_u, face_v)
            if res:
                closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
                out[e, i] = wp.length(q - closest) * sign
            else:
                out[e, i] = max_dist

        _wp = wp
        _KERNEL = _sdf_batched
    return _wp, _KERNEL


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Reorder a ``(..., 4)`` quaternion from the project's wxyz to Warp's xyzw."""
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion must be (..., 4), got {tuple(quat.shape)}")
    return quat[..., [1, 2, 3, 0]]


def goal_relative_transform(
    part_pos: torch.Tensor,
    part_quat: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (position, xyzw quaternion) in the seated-goal frame; input quaternions use wxyz."""
    rel_pos, rel_quat = im.relative_pose(goal_pos, goal_quat, part_pos, part_quat)
    return rel_pos, quat_wxyz_to_xyzw(rel_quat)


def clamp_outside(signed_dist: torch.Tensor) -> torch.Tensor:
    """Drop everything at or inside the isosurface, keep the outside distance."""
    return torch.where(signed_dist < 0.0, torch.zeros_like(signed_dist), signed_dist)


def sample_surface_points(obj_path: str, num_sample_points: int, seed: int):
    """Load an OBJ and sample ``num_sample_points`` points on its surface."""
    import numpy as np
    import trimesh

    if num_sample_points <= 0:
        raise ValueError(f"num_sample_points must be positive, got {num_sample_points}")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if not hasattr(mesh, "faces"):
        raise TypeError(f"{obj_path} did not load as a single mesh: {type(mesh)!r}")

    points, _ = trimesh.sample.sample_surface_even(mesh, num_sample_points, seed=seed)
    return np.asarray(points, dtype=np.float64), mesh


class SdfDistanceQuery:
    """The part's surface points against the part's own surface at the goal."""

    def __init__(
        self,
        obj_path: str,
        num_sample_points: int,
        max_dist: float,
        device: str,
        seed: int,
        mesh_obj_path: str | None = None,
    ):
        """``obj_path`` is sampled; ``mesh_obj_path`` is QUERIED."""
        wp, _ = _warp()

        self.obj_path = obj_path
        self.mesh_obj_path = obj_path if mesh_obj_path is None else mesh_obj_path
        self.max_dist = float(max_dist)
        self.seed = int(seed)

        points_np, mesh = sample_surface_points(obj_path, num_sample_points, seed)
        self.num_points = int(points_np.shape[0])
        self.num_requested = int(num_sample_points)
        self.num_vertices = int(mesh.vertices.shape[0])
        self.num_faces = int(mesh.faces.shape[0])
        self.extents = tuple(float(v) for v in mesh.extents)
        self.sampled_area_m2 = float(mesh.area)

        if self.mesh_obj_path == obj_path:
            query_mesh = mesh
        else:
            import trimesh

            if not os.path.isfile(self.mesh_obj_path):
                raise FileNotFoundError(f"mesh OBJ not found: {self.mesh_obj_path}")
            query_mesh = trimesh.load(self.mesh_obj_path, process=False, force="mesh")
            if not hasattr(query_mesh, "faces"):
                raise TypeError(
                    f"{self.mesh_obj_path} did not load as a single mesh: "
                    f"{type(query_mesh)!r}"
                )
        self.mesh_num_vertices = int(query_mesh.vertices.shape[0])
        self.mesh_num_faces = int(query_mesh.faces.shape[0])
        self.mesh_extents = tuple(float(v) for v in query_mesh.extents)
        self.mesh_bounds = (
            tuple(float(v) for v in query_mesh.bounds[0]),
            tuple(float(v) for v in query_mesh.bounds[1]),
        )

        self.wp_device = wp.get_preferred_device() if device == "auto" else device
        self.torch_device = str(self.wp_device)
        self.mesh = wp.Mesh(
            points=wp.array(query_mesh.vertices, dtype=wp.vec3, device=self.wp_device),
            indices=wp.array(
                query_mesh.faces.flatten(), dtype=wp.int32, device=self.wp_device
            ),
        )
        self.points = wp.array(points_np, dtype=wp.vec3, device=self.wp_device)

    def signed_distances(
        self, rel_pos: torch.Tensor, rel_quat_xyzw: torch.Tensor
    ) -> torch.Tensor:
        """``(E, 3)`` + ``(E, 4)`` xyzw -> ``(E, N)`` SIGNED distances."""
        wp, kernel = _warp()

        if rel_pos.ndim != 2 or rel_pos.shape[1] != 3:
            raise ValueError(f"rel_pos must be (E, 3), got {tuple(rel_pos.shape)}")
        if rel_quat_xyzw.shape != (rel_pos.shape[0], 4):
            raise ValueError(
                f"rel_quat_xyzw must be ({rel_pos.shape[0]}, 4), "
                f"got {tuple(rel_quat_xyzw.shape)}"
            )

        num_envs = int(rel_pos.shape[0])
        pos_wp = wp.from_torch(
            rel_pos.detach().to(device=self.torch_device, dtype=torch.float32).contiguous(),
            dtype=wp.vec3,
        )
        quat_wp = wp.from_torch(
            rel_quat_xyzw.detach()
            .to(device=self.torch_device, dtype=torch.float32)
            .contiguous(),
            dtype=wp.quat,
        )
        out = wp.zeros(
            (num_envs, self.num_points), dtype=wp.float32, device=self.wp_device
        )

        for name, arr in (
            ("xf_pos", pos_wp),
            ("xf_quat", quat_wp),
            ("points", self.points),
            ("out", out),
        ):
            if str(arr.device) != str(self.wp_device):
                raise RuntimeError(
                    f"device mismatch before launch: '{name}' is on "
                    f"{arr.device!s}, the launch is on {self.wp_device!s}. "
                    f"torch_device was resolved to {self.torch_device!r}."
                )

        wp.launch(
            kernel=kernel,
            dim=(num_envs, self.num_points),
            inputs=[self.mesh.id, self.points, pos_wp, quat_wp, self.max_dist, out],
            device=self.wp_device,
        )
        return wp.to_torch(out)

    def interpen_distances(
        self, rel_pos: torch.Tensor, rel_quat_xyzw: torch.Tensor
    ) -> torch.Tensor:
        """``(E, N)`` interpenetration in the SOURCE's convention (SAPU)."""
        return im.interpen_from_signed(self.signed_distances(rel_pos, rel_quat_xyzw))

    def mean_outside_distance(
        self, rel_pos: torch.Tensor, rel_quat_xyzw: torch.Tensor
    ) -> torch.Tensor:
        """``(E,)``: the mean OUTSIDE distance per env."""
        return clamp_outside(self.signed_distances(rel_pos, rel_quat_xyzw)).mean(dim=1)

    def describe(self) -> str:
        """One line for the startup report: what was loaded, and how much of it."""
        line = (
            f"[insertion_sdf] sampled mesh {os.path.basename(self.obj_path)}: "
            f"{self.num_vertices} vertices, {self.num_faces} faces, "
            f"extents {self.extents[0]:.4f} x {self.extents[1]:.4f} x "
            f"{self.extents[2]:.4f} m | sampled {self.num_points} of "
            f"{self.num_requested} requested, seed {self.seed}, "
            f"max_dist {self.max_dist} m"
        )
        if self.mesh_obj_path != self.obj_path:
            line += (
                f"\n[insertion_sdf] QUERIED mesh "
                f"{os.path.basename(self.mesh_obj_path)}: "
                f"{self.mesh_num_vertices} vertices, {self.mesh_num_faces} faces, "
                f"extents {self.mesh_extents[0]:.4f} x {self.mesh_extents[1]:.4f} "
                f"x {self.mesh_extents[2]:.4f} m"
            )
        return line

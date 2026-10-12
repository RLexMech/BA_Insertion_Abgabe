# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Warp half of the SDF reward for the UR5e insertion -- M2.2, D-109 point (3).

WHAT THIS FILE IS FOR
---------------------
D-109 (3) replaced the proxy's approach / depth / yaw / align terms with ONE
SDF-distance term: sample N points on the part's surface, move them to the
part's CURRENT pose, and measure their distance to the part's own surface at
the GOAL pose. One number carries position and rotation together, which is the
whole reason the decision picked it ("computes distance on a single manifold,
obviating tuning").

The published implementation is Isaac Lab 2.3.2,
``isaaclab_tasks/direct/automate/industreal_algo_utils.py``:
``load_asset_mesh_in_warp`` (L66-L89), ``mesh_sdf`` (L297-L306),
``get_batch_sdf`` (L314-L327) and ``get_sdf_reward`` (L97-L157), called from
``assembly_env.py:583``. This module follows that code and departs from it in
exactly two places, both named below. Nothing here is invented for its own
sake.

SAPU IS IN THIS FILE NOW, AND IT NEEDED NO SECOND KERNEL
---------------------------------------------------------
The SAPU / interpenetration query (``get_interpen_dist``, L344-L378) differs
from ``mesh_sdf`` by one line -- it keeps only the negative half -- so the port
is ``clamp(signed, max=0.0)`` on the result this file already produces. That is
``interpen_distances`` below, and the entry that settled it is
``docs/decisions_inbox.md``, "The SAPU port needs no second Warp kernel, and
the FIXTURE mesh is the only new thing it needs" (2026-08-30). Do not write a
second kernel.

This paragraph used to say the opposite -- that SAPU was deliberately left out
because the fixture surface is open and the sign would be unread. Half of that
still stands and is a NAMED GAP rather than a solved problem:
``wp.mesh_query_point`` documents "Mesh must be watertight!", the fixture
carries unpaired edges, and RT-90 measured the sign only FAR OUTSIDE the part
mesh, never at the surface. What closes it is not an argument but the LOCATION
of those edges against the volume the part sweeps --
``scripts/check_fixture_mesh.py`` (RT-91). Until that run is judged, every SAPU
number out of this file is UNVERIFIED.

The reward MAPPING is not here either. The SDF distance goes through
``insertion_math.squash`` (D-109 (6)), which already exists and is checked
offline. This file produces the DISTANCE and nothing else.

DEPARTURE 1 -- BATCHED, NOT A PYTHON LOOP OVER ENVS
---------------------------------------------------
``get_sdf_reward`` loops over envs and calls ``mesh_copy.refit()`` inside the
loop -- a BVH rebuild per env per step. It has to, because it moves a copy of
the part mesh to each env's goal pose.

We do not. A distance is invariant when the same rigid transform is applied to
both sides, so applying ``T_goal^-1`` to both turns "points at ``T_curr``
against a mesh at ``T_goal``" into "points at ``T_goal^-1 . T_curr`` against a
mesh at the IDENTITY" -- the same number, against a mesh built and refitted
ONCE. ``insertion_math.relative_pose`` computes that composition and is
already green offline.

This SUBSTITUTION is ours; the algebra is standard rigid-body composition, but
no source performs it. It is exact rather than an approximation. The P4 gate
still stands: the batched result must be compared against a per-env reference
run on the training PC before the SDF gates anything.

DEPARTURE 2 -- THE QUATERNION IS CONVERTED. THE PUBLISHED CODE DOES NOT
-----------------------------------------------------------------------
``wp.transform(p, q)`` takes ``q`` as **xyzw**. Isaac Lab states this about its
own Warp usage in two places and acts on it in a third:

* ``isaaclab/utils/warp/fabric.py:60`` -- "(Warp uses xyzw, Isaac Lab uses
  wxyz)."
* ``isaaclab/utils/warp/fabric.py:112`` -- "It handles the quaternion
  convention conversion (Isaac Lab uses wxyz, Warp uses xyzw)."
* ``isaaclab/utils/warp/ops.py:336-339`` -- ``convert_quat(..., "xyzw")``
  immediately before ``wp.from_torch(..., dtype=wp.quat)``.

``get_sdf_reward`` receives ``self.held_quat = self._held_asset.data.
root_quat_w`` (``assembly_env.py:280``), which is Isaac Lab **wxyz**, and feeds
it into ``wp.transform`` with NO conversion (``industreal_algo_utils.py`` L121,
L136). Reading ``(w,x,y,z)`` as ``(x,y,z,w)`` turns the identity ``(1,0,0,0)``
into a 180 deg rotation about x, and the two sides of the distance are rotated
differently, so the error does not cancel.

WE DO NOT COPY THAT. ``goal_relative_transform`` below emits xyzw, once, at the
single point where torch hands over to Warp. This is a DEVIATION FROM THE
SOURCE, recorded as one rather than silently fixed. It is read from the Isaac
Lab source and has NOT been confirmed against a running Warp -- the probe in
``scripts/check_insertion_sdf.py`` is what will confirm it.

THE OPEN QUESTION THIS FILE EXISTS TO SETTLE
--------------------------------------------
``mesh_sdf`` multiplies the distance by ``sign`` from ``wp.mesh_query_point``,
and that call is documented in the same file as ``NOTE: Mesh must be
watertight!`` (L356-L358). RT-87 measured our part mesh and found a floor of
380 unpaired edges -- but only at a weld tolerance of 1.8 nm, because the
mesh's own closest distinct pair is 0.00008 mm apart. So the part is NOT PROVEN
closed and NOT PROVEN open in any band that matters.

That question is settled by MEASUREMENT, not by choosing a route.
``scripts/check_insertion_sdf.py`` translates the sampled points by offsets
larger than the mesh's own extent -- where every point is outside by
construction, with no oracle needed -- and asserts that not one distance comes
back negative. If the sign holds on our actual mesh, the open-mesh worry is
empirically dead for this term.

API NOTES, stated rather than assumed
-------------------------------------
Warp is NOT installed on the laptop, so nothing here has been executed against
it. Every Warp construct used below was copied from a call that ships in Isaac
Lab 2.3.2, never from memory:

* ``wp.transform(pos_array[i], quat_array[i])`` inside a kernel --
  ``isaaclab/utils/warp/kernels.py:218``.
* ``wp.from_torch(x, dtype=wp.quat)`` after an xyzw conversion --
  ``isaaclab/utils/warp/ops.py:336-339``; ``dtype=wp.vec3f`` --
  ``isaaclab/assets/articulation/articulation.py:1071``.
* ``wp.array2d(dtype=wp.float32)`` as a kernel parameter and a multi-value
  ``wp.tid()`` -- ``kernels.py:84-91``, ``kernels.py:133``.
* The SEVEN-ARGUMENT ``wp.mesh_query_point(mesh, q, max_dist, sign,
  face_index, face_u, face_v)`` returning a bool, plus ``wp.mesh_eval_position``
  -- ``industreal_algo_utils.py:302-304``. The newer struct-returning form
  appears in ``kernels.py`` for RAY queries; whether the POINT query offers the
  same shape in the Warp inside Isaac Sim 5.1.0 is unchecked. If the call is
  rejected, the training PC says so at once and the installed Warp is read
  then -- a measurement, not a guess.

``trimesh.sample.sample_surface_even`` needs ``scipy`` (it calls
``trimesh.points.remove_close``), and scipy is not on the laptop. The sampling
half therefore runs on the training PC only; the pure-torch half above it runs
anywhere.

IT OWNS NO NUMBERS
------------------
Sample count, ``max_dist``, seed and device all arrive as arguments, the same
rule ``insertion_math.py`` follows. ``num_sample_points`` HAS a source, and
since 2026-08-30 it is ours rather than borrowed: **D-154 sets it to 64,000**,
measured against the cost curve RT-97 produced. Isaac Lab's
``num_mesh_sample_points = 1000`` was chosen for another task's env count and
is superseded for this task. The caller still states it and every run prints
it -- this module owns no numbers.

Warp is imported LAZILY, inside ``_warp()``. That is what lets the laptop
import this module and check the torch half offline.
"""

from __future__ import annotations

import os

import torch

from . import insertion_math as im

# Filled by _warp() on first use. A module-level import would make this file
# unimportable on the laptop, and then nothing in it could be checked offline.
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
            """Signed distance of every sampled point, for every env.

            One thread per (env, point). ``points`` is the SINGLE sampled set in
            the mesh's own frame; ``xf_pos[e]`` / ``xf_quat[e]`` are env e's
            ``T_goal^-1 . T_curr``, the quaternion already in Warp's xyzw order.
            The mesh stays at the identity, so no copy and no ``refit()``
            happens here -- departure 1 of the module docstring.

            The body is ``mesh_sdf`` (industreal_algo_utils.py L297-L306) with
            the env loop lifted into the thread index. Sign and fallback are the
            source's, unchanged: outside positive, inside negative, and a point
            with no surface within ``max_dist`` reports ``max_dist``.
            """
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


# ===========================================================================
#  Pure torch -- no Warp, no trimesh, no scipy. Checked offline.
# ===========================================================================


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Reorder a ``(..., 4)`` quaternion from the project's wxyz to Warp's xyzw.

    The ONLY place in this stream where a quaternion changes convention, and
    departure 2 of the module docstring. Written as an index gather rather than
    a cat of slices, so a wrong order shows up as a wrong list instead of as a
    swapped argument nobody rereads.
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion must be (..., 4), got {tuple(quat.shape)}")
    return quat[..., [1, 2, 3, 0]]


def goal_relative_transform(
    part_pos: torch.Tensor,
    part_quat: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``T_goal^-1 . T_curr`` per env, ready for Warp.

    Inputs are ``(E, 3)`` positions and ``(E, 4)`` **wxyz** quaternions.
    Returns ``(rel_pos, rel_quat_xyzw)`` -- ``(E, 3)`` and ``(E, 4)``, split
    the way ``wp.from_torch`` consumes them (``ops.py:336-339``).

    The composition itself is ``insertion_math.relative_pose`` and is not
    re-derived here.
    """
    rel_pos, rel_quat = im.relative_pose(goal_pos, goal_quat, part_pos, part_quat)
    return rel_pos, quat_wxyz_to_xyzw(rel_quat)


def clamp_outside(signed_dist: torch.Tensor) -> torch.Tensor:
    """Drop everything at or inside the isosurface, keep the outside distance.

    ``get_sdf_reward`` L155: ``torch.where(sdf_dist < 0.0, 0.0, sdf_dist)``.
    Copied in behaviour, not paraphrased: a NEGATIVE distance means the query
    point sits inside the part at its goal pose, which the reward neither pays
    for nor punishes.

    THE EXPLOIT THIS INVITES, answered per the project rule: if the sign is
    wrong on an open mesh, a point that is genuinely far OUTSIDE reads negative
    and is clamped to zero, so a bad pose scores like a seated one. That is
    exactly why the sign is probed on the real mesh before this term gates
    anything.
    """
    return torch.where(signed_dist < 0.0, torch.zeros_like(signed_dist), signed_dist)


# ===========================================================================
#  trimesh half -- training PC only (sample_surface_even needs scipy).
# ===========================================================================


def sample_surface_points(obj_path: str, num_sample_points: int, seed: int):
    """Load an OBJ and sample ``num_sample_points`` points on its surface.

    Returns ``(points, mesh)`` -- an ``(N, 3)`` numpy array and the loaded
    ``trimesh.Trimesh``. The mesh comes back because the caller needs its
    vertices, faces and extents; loading it twice would be two chances to load
    two different files.

    Pattern and sampler are ``load_asset_mesh_in_warp`` (L69, L81):
    ``trimesh.load`` then ``trimesh.sample.sample_surface_even``. The SEED is
    ours and the source has none -- an unseeded sample makes two runs of the
    same policy disagree by construction, and the project rule is that the seed
    is always set. ``seed=`` is a real keyword of
    ``sample_surface_even(mesh, count, radius=None, seed=None)``, checked
    against the installed trimesh 4.12.2.

    ``sample_surface_even`` may return FEWER than ``num_sample_points`` points
    -- it drops points that fall too close together -- which the source neither
    says nor handles. The count that came back is returned as-is and the caller
    reports it; padding it would invent points.
    """
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


# ===========================================================================
#  The query object
# ===========================================================================


class SdfDistanceQuery:
    """The part's surface points against the part's own surface at the goal.

    Built ONCE per env instance. Holds one ``wp.Mesh`` at the identity and one
    sampled point set; neither is rebuilt, moved or refitted afterwards -- that
    is what departure 1 buys.

    Every number arrives as an argument. ``max_dist`` is the source's ``1.5``
    (``mesh_sdf`` L299, ``get_batch_sdf`` L323) in METRES: the radius beyond
    which the query gives up and reports ``max_dist`` instead of a distance, so
    it has to exceed any distance the reward should still see.
    """

    def __init__(
        self,
        obj_path: str,
        num_sample_points: int,
        max_dist: float,
        device: str,
        seed: int,
        mesh_obj_path: str | None = None,
    ):
        """``obj_path`` is sampled; ``mesh_obj_path`` is QUERIED.

        They are the same file for the SDF reward term -- the part against its
        own surface at the goal pose -- and that is the default, so nothing
        that existed before this argument changes.

        They differ for SAPU (D-109 point (10)): the PART's sample points are
        queried against the FIXTURE mesh, which is what interpenetration
        means. Splitting the two paths here rather than writing a second query
        class is the whole port -- see ``insertion_math.interpen_from_signed``
        for why no second Warp kernel is needed either.

        The transform argument then carries the other frame: ``T_goal^-1 .
        T_curr`` for the reward term, ``T_fixture^-1 . T_part`` for SAPU. Both
        are ``goal_relative_transform``; only what is passed as "goal"
        changes, and the mesh sits at the identity either way.
        """
        wp, _ = _warp()

        self.obj_path = obj_path
        self.mesh_obj_path = obj_path if mesh_obj_path is None else mesh_obj_path
        self.max_dist = float(max_dist)
        self.seed = int(seed)

        points_np, mesh = sample_surface_points(obj_path, num_sample_points, seed)
        self.num_points = int(points_np.shape[0])
        self.num_requested = int(num_sample_points)
        # These three describe the SAMPLED mesh, not the queried one. The sign
        # probe's ladder translates the points by their OWN extents, so this
        # must not silently become the fixture's when the two differ.
        self.num_vertices = int(mesh.vertices.shape[0])
        self.num_faces = int(mesh.faces.shape[0])
        self.extents = tuple(float(v) for v in mesh.extents)
        # The sampled mesh's REAL surface area, square metres. It exists so the
        # point spacing can be stated against the surface the points actually
        # lie on: D-122 computed the spacing from a BOX around the part and
        # said in the same breath that the box is a LOWER bound on the area,
        # because the real part carries two lugs and a stepped underside. This
        # attribute replaces that bound with the measurement. Nothing in the
        # query reads it; it is reported, never used in the reward path.
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
        # The QUERIED mesh's own extents, kept beside its counts and separate
        # from `self.extents` above. The sign probe's ladder translates the
        # sampled points until they are outside BY CONSTRUCTION, and "outside"
        # is a statement about the mesh being queried. With the two meshes the
        # same (the SDF reward) the two tuples are equal; with them different
        # (SAPU) the part's extents would move the cloud far too little to
        # clear the fixture, and the probe would assert nothing.
        self.mesh_extents = tuple(float(v) for v in query_mesh.extents)
        # And its POSITION, not only its size. Extents alone cannot say whether
        # a translated point cloud has left the mesh: two boxes of the same
        # span sitting a metre apart and two sitting on top of each other have
        # identical extents. The sign probe needs "provably outside", and that
        # is a statement about bounds.
        self.mesh_bounds = (
            tuple(float(v) for v in query_mesh.bounds[0]),
            tuple(float(v) for v in query_mesh.bounds[1]),
        )

        self.wp_device = wp.get_preferred_device() if device == "auto" else device
        # The SAME device, spelled the way torch spells it. Everything the
        # kernel reads has to live here: RT-88 launched on cuda:0 with query
        # arrays still on the host, and the run did not come back.
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
        """``(E, 3)`` + ``(E, 4)`` xyzw -> ``(E, N)`` SIGNED distances.

        Signed, not clamped: the caller decides, and the sign probe has to see
        the raw value. Build the inputs with ``goal_relative_transform``.
        """
        wp, kernel = _warp()

        if rel_pos.ndim != 2 or rel_pos.shape[1] != 3:
            raise ValueError(f"rel_pos must be (E, 3), got {tuple(rel_pos.shape)}")
        if rel_quat_xyzw.shape != (rel_pos.shape[0], 4):
            raise ValueError(
                f"rel_quat_xyzw must be ({rel_pos.shape[0]}, 4), "
                f"got {tuple(rel_quat_xyzw.shape)}"
            )

        num_envs = int(rel_pos.shape[0])
        # DEVICE, not just dtype. `.to(torch.float32)` alone changes the dtype
        # and leaves a host tensor on the host, which is what RT-88 did on
        # 2026-08-29: the mesh and the sampled points sat on cuda:0, the query
        # transforms sat on the CPU, and `wp.launch(device=cuda:0)` was handed
        # host pointers.
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

        # A LOUD FAILURE INSTEAD OF A SILENT ONE. Whether Warp refuses a
        # cross-device launch or reads the host pointer as a device pointer is
        # not something this stream has measured, and the second outcome is an
        # illegal access that can take the CUDA context (and the process) with
        # it rather than raising. This guard makes the mismatch a Python error
        # either way, and it also catches `str(wp_device)` not being a torch
        # device name -- an API shape nothing here has verified.
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
        """``(E, N)`` interpenetration in the SOURCE's convention (SAPU).

        Every element is ``<= 0``: zero where the sample point clears the
        queried mesh, negative where it is inside. Feed
        ``insertion_math.max_interpen_dist`` with this.

        Only meaningful on a query built with the FIXTURE as
        ``mesh_obj_path``, and with ``T_fixture^-1 . T_part`` as the
        transform. It is a method rather than a call-site clamp so there is
        exactly one right way to build the array -- clamping the wrong end
        keeps the clearances and drops the penetrations, and the part then
        reads as one that never tunnels.
        """
        return im.interpen_from_signed(self.signed_distances(rel_pos, rel_quat_xyzw))

    def mean_outside_distance(
        self, rel_pos: torch.Tensor, rel_quat_xyzw: torch.Tensor
    ) -> torch.Tensor:
        """``(E,)``: the mean OUTSIDE distance per env.

        ``get_sdf_reward`` L155-L157 exactly -- clamp the inside away, then take
        the mean over the sampled points. What the reward finally pays is this
        distance through ``insertion_math.squash`` (D-109 (6)), which is not
        applied here.
        """
        return clamp_outside(self.signed_distances(rel_pos, rel_quat_xyzw)).mean(dim=1)

    def describe(self) -> str:
        """One line for the startup report: what was loaded, and how much of it.

        A SECOND line appears only when the queried mesh differs from the
        sampled one, i.e. in the SAPU configuration. Printing it
        unconditionally would put the same numbers twice under two names in
        every SDF-reward log, and a reader who sees one line knows the two
        meshes are the same file.
        """
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

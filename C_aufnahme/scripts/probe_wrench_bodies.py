# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Which body carries the force signal? Measure it, do not argue it.

THE QUESTION. D-114 puts three force components into the observation and one
force threshold into the termination, both read from
``ArticulationData.body_incoming_joint_wrench_b`` (FORGE pattern,
``forge_env.py:95``, NOT a ContactSensor). Branch ``p2-rl-code`` has the
config field ``force_sensor_body_name`` waiting and refuses a placeholder --
correctly, since a zero that looks like a reading is the same class of silent
defect as the SAPU sign error that branch just found.

D-114 says the wrench is read "at a dedicated force-sensor link", and
``HANDOFF-SZENE.md`` assigns building that link to this stream as an asset
job. But our tool is ALREADY its own link: ``tool_link``, joined to
``wrist_3_link`` by the fixed joint ``tool_weld`` (D-076..D-079). FORGE needs
a dedicated link because its held asset is not welded to the arm; ours is.
If the wrench incoming at ``tool_link`` is the flange-to-tool wrench, then an
asset re-authoring collapses into one string.

That is an api-shape claim in the sense of D-080 and it is paid for with a
probe, not with the paragraph above.

THE CONTROL, and it is what makes this a measurement rather than a printout.
At rest under gravity every link of a serial chain carries a non-zero
incoming joint wrench: it holds up everything distal to it. ``tool_link`` is
the LAST link, so nothing hangs below it, and the force incoming at its joint
must be its own weight:

    |f| ~= m_tool * g

``m_tool`` is read from the articulation itself, not supplied here. A reading
that matches its own authored mass is evidence the channel means what D-114
assumes. A reading that does not is the finding, and it is worth more than a
guess about which link to use.

Both are results. Neither is assumed.

Gravity is ON in this scene (D-105, verified RT-45/RT-46), which is what
makes the control available at all.

Nothing is written. No file is modified. The env is built, reset, stepped a
few times so the solver settles, and read.

UNVERIFIED -- authored on the dev laptop, which has no Isaac. On the training
machine:

    conda activate env_isaaclab
    python scripts\\probe_wrench_bodies.py --headless

PRUEFBLOCK B (D-188, 2026-09-13): ``--tare-check``
---------------------------------------------------
The torque twin of the control above, in ONE bundled run through the env's
own ``step()`` path (4 envs, ``obs_wrench_mode=wrench``, every sigma 0,
fixture noise off), four points, each a measurement with a prediction
written next to it:

1. REST. Zero actions for ``--settle-steps``. The raw torque at the force
   link against ``r x F_hold`` for TWO candidate reference points -- the
   parent link origin (the Isaac test's, ``test_articulation.py:1871-1882``)
   and the child link origin -- and BOTH signs. The pair whose residual is
   within tolerance names the reference point and the sign convention;
   none within tolerance is a finding, not a pass.
2. KNOWN LOAD. ``--load-n`` newtons along the tool's local +x at the tool
   tip (``_tip_offset_local``), through ``set_external_force_and_torque``
   (local link frame). Prediction: delta force ``R_p^T R_c F``, delta torque
   ``R_p^T (r_tip x R_c F)`` with ``r_tip`` from the reference point to the
   tip. Read at the raw wrench, the tared buffers, the EMA AND channels 28:31
   of the policy observation -- frame, sign, reference point and wiring in
   one number each.
3. PARTIAL RESET. ``_reset_idx([0])`` with the load still on, one step:
   env 0's buffers are 0 and ``_wrench_valid[0]`` False on that step, True
   on the next; env 1's torque is unchanged.
4. DOUBLE READ. ``_get_observations()`` twice without physics: the EMA
   buffers are byte-identical.

Tolerances are OWN CHOICES, stated here rather than tuned: rest residual
max(0.005 N m, 1 % of the prediction) -- RT-180c's tared force was
0.00036 N of 8.08, i.e. 4.5e-5 relative, so 1 % is 200x that; load
magnitude ratio in [0.97, 1.03] and direction cosine >= 0.999 (2.6 deg).
5 N is "well above the force precision, well below the 50 N abort" -- own
construction, named. Writes ``wrench_tare_check.json``. Exit 0 only when
all four points pass.

    python scripts\\probe_wrench_bodies.py --tare-check --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import traceback

from isaaclab.app import AppLauncher

SCRIPT_MARKER = "probe_wrench_bodies-2026-09-13f"

parser = argparse.ArgumentParser(description="Probe body_incoming_joint_wrench_b per body (D-114 route).")
parser.add_argument("--settle-steps", type=int, default=30,
                    help="Physics steps before reading, so the solver settles under gravity.")
parser.add_argument("--json", type=str, default=None, help="Metrics output path.")
parser.add_argument("--tare-check", action="store_true",
                    help="Pruefblock B (D-188): rest, known load, partial reset, double read.")
parser.add_argument("--num-envs", type=int, default=4, help="--tare-check only.")
parser.add_argument("--load-n", type=float, default=5.0,
                    help="--tare-check only: the known load, N along the tool's local +x at the tip.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"[wrench] marker: {SCRIPT_MARKER}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from insertion.tasks.direct.insertion.insertion_env import InsertionEnv  # noqa: E402
from insertion.tasks.direct.insertion.insertion_env_cfg import InsertionEnvCfg  # noqa: E402
from insertion.tasks.direct.insertion import insertion_tasks_cfg as geom  # noqa: E402
from insertion.tasks.direct.insertion import insertion_math  # noqa: E402

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

G = 9.81


def read_attr(data, name):
    """Return an attribute or None. An absent field is a FINDING, not a crash.

    A missing attribute here costs a whole copy-and-run round trip if it ends
    the script; printing its name is worth more than a traceback.
    """
    try:
        return getattr(data, name)
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        print(f"[wrench] WARNING: '{name}' unavailable ({type(exc).__name__}: {exc})")
        return None


def main_bodies() -> int:
    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = 1
    cfg.fixture_pos_noise_xy = 0.0
    cfg.fixture_yaw_noise_rad = 0.0
    env = InsertionEnv(cfg)
    robot = env.robot

    result: dict = {"marker": SCRIPT_MARKER, "tool_link_name": geom.TOOL_LINK_NAME}

    env.reset()
    for _ in range(args_cli.settle_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.cfg.sim.dt)

    names = list(robot.body_names)
    result["body_names"] = names
    print(f"[wrench] {len(names)} bodies: {names}")

    wrench = read_attr(robot.data, "body_incoming_joint_wrench_b")
    if wrench is None:
        result["verdict"] = "CHANNEL_ABSENT"
        result["reason"] = ("body_incoming_joint_wrench_b does not exist on this "
                            "ArticulationData. D-114's route is not available as assumed; "
                            "the version or the field name has to be settled before anything "
                            "reads a force.")
        print(f"[wrench] {result['reason']}")
        return finish(result, env, ok=False)

    masses = read_attr(robot.data, "default_mass")
    print(f"[wrench] wrench tensor shape: {tuple(wrench.shape)}")
    result["wrench_shape"] = list(wrench.shape)

    # --- per body -----------------------------------------------------------
    print(f"[wrench] {'idx':>3} {'body':<24} {'|f| [N]':>10} {'|tau| [Nm]':>11} "
          f"{'mass [kg]':>10} {'m*g [N]':>9}")
    rows = []
    for i, name in enumerate(names):
        w = wrench[0, i]
        f_norm = float(torch.linalg.norm(w[:3]).item())
        t_norm = float(torch.linalg.norm(w[3:]).item())
        m = None if masses is None else float(masses[0, i].item())
        mg = None if m is None else m * G
        rows.append({"index": i, "body": name, "force_norm_n": f_norm,
                     "torque_norm_nm": t_norm, "mass_kg": m, "weight_n": mg,
                     "force_xyz_n": [float(v) for v in w[:3]]})
        print(f"[wrench] {i:>3} {name:<24} {f_norm:>10.4f} {t_norm:>11.4f} "
              f"{'--' if m is None else f'{m:>10.5f}'} {'--' if mg is None else f'{mg:>9.4f}'}")
    result["bodies"] = rows

    nonzero = [r["body"] for r in rows if r["force_norm_n"] > 1e-6]
    result["nonzero_force_bodies"] = nonzero
    print(f"[wrench] bodies with a non-zero force: {len(nonzero)} of {len(rows)}")

    # --- the control --------------------------------------------------------
    tool = next((r for r in rows if r["body"] == geom.TOOL_LINK_NAME), None)
    if tool is None:
        result["verdict"] = "TOOL_LINK_ABSENT"
        result["reason"] = (f"body '{geom.TOOL_LINK_NAME}' is not in the articulation; "
                            f"the weld did not survive authoring. bodies: {names}")
        print(f"[wrench] {result['reason']}")
        return finish(result, env, ok=False)

    result["tool_link_index"] = tool["index"]
    if tool["mass_kg"] is None:
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = ("the wrench reads, but default_mass is unavailable, so the "
                            "weight control cannot run. A reading without its control is "
                            "not evidence about what the channel means.")
        print(f"[wrench] {result['reason']}")
        return finish(result, env, ok=False)

    ratio = tool["force_norm_n"] / tool["weight_n"] if tool["weight_n"] > 0 else float("inf")
    result["tool_force_over_weight"] = ratio
    print(f"[wrench] CONTROL: {geom.TOOL_LINK_NAME} reads |f| = {tool['force_norm_n']:.4f} N "
          f"against its own weight {tool['weight_n']:.4f} N (mass {tool['mass_kg']:.5f} kg) "
          f"-> ratio {ratio:.3f}")

    # 10 % covers solver residual and the small dynamic term of a settling arm.
    # It is a band around a PREDICTED value, not a tuned threshold: the
    # prediction is m*g and the only question is whether the reading is it.
    if 0.9 <= ratio <= 1.1:
        result["verdict"] = "TOOL_LINK_CARRIES_THE_WRENCH"
        result["reason"] = (
            f"the force incoming at '{geom.TOOL_LINK_NAME}' matches its own authored weight "
            f"to {abs(1-ratio)*100:.1f} %. It is the last link, so nothing hangs below it and "
            f"m*g is the whole prediction. The channel therefore reads the flange-to-tool "
            f"wrench, and force_sensor_body_name = '{geom.TOOL_LINK_NAME}' is a measured "
            f"answer, not a guess. No dedicated force-sensor link has to be authored."
        )
    else:
        result["verdict"] = "UNEXPECTED_READING"
        result["reason"] = (
            f"'{geom.TOOL_LINK_NAME}' reads {ratio:.3f}x its own weight, outside the 0.9..1.1 "
            f"band. The channel exists but does not mean what D-114 assumes at this link. Do "
            f"NOT set force_sensor_body_name from this run -- the per-body table above is the "
            f"finding, and the dedicated-link route stays open."
        )
    print(f"[wrench] VERDICT: {result['verdict']}")
    print(f"[wrench] {result['reason']}")
    return finish(result, env, ok=result["verdict"] == "TOOL_LINK_CARRIES_THE_WRENCH")


def _l(t) -> list:
    """Tensor -> nested Python list for the JSON."""
    return t.detach().cpu().tolist()


def _rot_t(quat_w: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    """World vector (N, 3) -> the body frame of ``quat_w`` (N, 4): R^T v,
    through the math module's own convention (axes are COLUMNS)."""
    return insertion_math.rotate_into_frame(insertion_math.axes_from_quat(quat_w), v_w)


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row direction cosine of two (N, 3) batches; 0 where either is ~0."""
    na = torch.linalg.norm(a, dim=-1)
    nb = torch.linalg.norm(b, dim=-1)
    dot = (a * b).sum(dim=-1)
    return torch.where((na > 1e-9) & (nb > 1e-9), dot / (na * nb).clamp_min(1e-12),
                       torch.zeros_like(dot))


def main_tare_check() -> int:
    """Pruefblock B (D-188). See the module docstring for the four points."""
    # Own tolerances, named in the docstring.
    REST_ABS_NM, REST_REL = 0.005, 0.01
    LOAD_RATIO = (0.97, 1.03)
    LOAD_COS = 0.999

    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.seed = 42
    cfg.obs_wrench_mode = "wrench"
    # A MEASUREMENT: the observation must BE the reading (zero_agent.py's
    # pinning, same reason), and one deterministic fixture pose.
    cfg.obs_noise_pocket_pos_std_m = 0.0
    cfg.force_obs_noise_std_n = 0.0
    cfg.torque_obs_noise_std_nm = 0.0
    cfg.grasp_obs_offset_x_m = 0.0
    cfg.fixture_pos_noise_xy = 0.0
    cfg.fixture_yaw_noise_rad = 0.0
    cfg.fixture_tilt_noise_rad = 0.0
    cfg.report_at_steps = (2,)  # one startup report, not four (log length)
    # JOINT_PD FOR THE LOAD PROBE (RT-192a4 finding): under the OSC a zero
    # action re-anchors the target to the CURRENT pose every physics step,
    # so a constant load drags the arm away without limit (a2: 67 deg, 0.1 m)
    # and the before/after wrench carries the pose change. Overwriting the
    # joint state each substep (a4) is no equilibrium either -- the force
    # ratio fell to 0.33. The USD drives of joint_pd HOLD the pose with
    # stiffness: a small, static deflection and a real reaction wrench. The
    # wrench read, the tare and the obs wiring are the same code in both
    # modes; only the actuator differs.
    cfg.control_mode = "joint_pd"
    env = InsertionEnv(cfg)
    robot = env.robot
    n = env.num_envs
    dev = env.device
    result: dict = {"marker": SCRIPT_MARKER, "mode": "tare-check", "num_envs": n,
                    "settle_steps": int(args_cli.settle_steps), "load_n": float(args_cli.load_n),
                    "tolerances": {"rest_abs_nm": REST_ABS_NM, "rest_rel": REST_REL,
                                   "load_ratio": list(LOAD_RATIO), "load_cos": LOAD_COS}}
    tool = env._force_body_idx
    parent = env._force_parent_idx
    if tool is None or parent is None or env._hold_force_w is None:
        result["verdict"] = "NO_TARE_WIRING"
        result["reason"] = (f"force body {tool}, parent {parent}, hold force {env._hold_force_w}: "
                            "the tare is not wired, nothing to check.")
        print(f"[tare] {result['reason']}")
        return finish(result, env, ok=False)
    names = list(robot.body_names)
    result["force_body"] = names[tool]
    result["parent_body"] = names[parent]
    hold_w = env._hold_force_w.reshape(1, 3).repeat(n, 1)
    q_lo, q_hi = insertion_math.obs_slices("wrench")["torque"]
    f_lo, f_hi = insertion_math.obs_slices("wrench")["force"]
    zero = torch.zeros(n, int(env.cfg.action_space), device=dev)

    def step():
        return env.step(zero)[0]["policy"]

    def read() -> dict:
        d = robot.data
        return {
            "raw": d.body_incoming_joint_wrench_b[:, tool, 0:6].clone(),
            "f_tared": env._force_tared_raw.clone(),
            "t_tared": env._torque_tared_raw.clone(),
            "f_ema": env._force_smooth.clone(),
            "t_ema": env._torque_smooth.clone(),
            "valid": env._wrench_valid.clone(),
            "com_w": d.body_com_pos_w[:, tool].clone(),
            "pos_tool": d.body_pos_w[:, tool].clone(),
            "pos_parent": d.body_pos_w[:, parent].clone(),
            "q_tool": d.body_quat_w[:, tool].clone(),
            "q_parent": d.body_quat_w[:, parent].clone(),
        }

    passed: dict = {}
    env.reset()

    # ---- 1. REST ---------------------------------------------------------
    for _ in range(int(args_cli.settle_steps)):
        obs_rest = step()
    r1 = read()
    tau_raw = r1["raw"][:, 3:6]
    f_raw = r1["raw"][:, 0:3]
    cands = {
        "parent_origin": r1["com_w"] - r1["pos_parent"],
        "child_origin": r1["com_w"] - r1["pos_tool"],
    }
    p1 = {"force_raw_n": _l(f_raw), "force_raw_norm_n": _l(torch.linalg.norm(f_raw, dim=-1)),
          "hold_norm_n": float(torch.linalg.norm(env._hold_force_w)),
          "force_tared_norm_n": _l(torch.linalg.norm(r1["f_tared"], dim=-1)),
          "torque_raw_nm": _l(tau_raw), "torque_raw_norm_nm": _l(torch.linalg.norm(tau_raw, dim=-1)),
          "torque_tared_env_nm": _l(r1["t_tared"]),
          "torque_tared_env_norm_nm": _l(torch.linalg.norm(r1["t_tared"], dim=-1)),
          "candidates": {}}
    fits = []
    print(f"[tare] 1 REST after {args_cli.settle_steps} steps: |f_raw| "
          f"{[round(v, 4) for v in p1['force_raw_norm_n']]} N vs hold {p1['hold_norm_n']:.4f} N; "
          f"|f_tared| {[round(v, 5) for v in p1['force_tared_norm_n']]} N")
    for name, lever_w in cands.items():
        pred = _rot_t(r1["q_parent"], torch.cross(lever_w, hold_w, dim=-1))
        pn = torch.linalg.norm(pred, dim=-1)
        for sign, sgn in (("+", 1.0), ("-", -1.0)):
            resid = torch.linalg.norm(tau_raw - sgn * pred, dim=-1)
            tol = torch.clamp(REST_REL * pn, min=REST_ABS_NM)
            ok = bool((resid <= tol).all())
            key = f"{name}{sign}"
            p1["candidates"][key] = {"pred_nm": _l(sgn * pred), "pred_norm_nm": _l(pn),
                                     "residual_nm": _l(resid), "tol_nm": _l(tol), "fits": ok}
            print(f"[tare]   ref {name} sign {sign}: |pred| {[round(v, 4) for v in _l(pn)]} Nm, "
                  f"residual {[round(v, 5) for v in _l(resid)]} Nm, tol "
                  f"{[round(v, 5) for v in _l(tol)]} -> {'FITS' if ok else 'no'}")
            if ok:
                fits.append((name, sgn))
    p1["fits"] = [f"{a}{'+' if s > 0 else '-'}" for a, s in fits]
    # The env's own tare uses the parent origin with the + sign; it is right
    # exactly when that pair fits, and the tared buffer then reads ~0.
    env_tare_ok = ("parent_origin", 1.0) in fits and bool(
        (torch.linalg.norm(r1["t_tared"], dim=-1) <= torch.clamp(
            REST_REL * torch.linalg.norm(cands["parent_origin"], dim=-1) * p1["hold_norm_n"],
            min=REST_ABS_NM)).all())
    p1["env_tare_fits"] = env_tare_ok
    # MEASURED RT-192a first run: the parent and child origins COINCIDE (the
    # weld joint sits at the link origin), so at rest both candidates fit and
    # this point cannot separate them; it pins the SIGN (+) and the tare
    # (~1e-6 N m residual). The reference point is point 2's job.
    p1["origins_distance_m"] = _l(torch.linalg.norm(r1["pos_tool"] - r1["pos_parent"], dim=-1))
    passed["1_rest"] = ("parent_origin", 1.0) in fits and ("parent_origin", -1.0) not in fits and env_tare_ok
    print(f"[tare] 1 RESULT: fitting (reference, sign) pairs {p1['fits']}; env tare (parent_origin+) "
          f"{'reads ~0' if env_tare_ok else 'does NOT read ~0'} -> {'PASS' if passed['1_rest'] else 'FAIL'}")
    result["1_rest"] = p1

    # ---- 2. KNOWN LOAD, TWO APPLICATION POINTS (RT-192a rerun, 2026-09-13) --
    # THE SIGN CONVENTION IS MEASURED, NOT ASSUMED, since the first RT-192a:
    # the incoming wrench is the PARENT's reaction, i.e. MINUS the load, for
    # the force (delta = -R_p^T F_w, ratio 1.0004, cos -1.000) exactly as for
    # gravity (raw = +hold = -m g). The prediction below carries that sign.
    # THE REFERENCE POINT is what the first run could not name: the lever
    # came out 0.255 m where the tip sits 0.152 m from the link origin. One
    # position cannot separate "positions are ignored" from "the moment is
    # taken about another point", so the SAME load goes to TWO points along
    # the tool axis; the DIFFERENCE of the two moments must be F * dz if the
    # application point is honoured, and the intercept names the reference
    # point as a distance d along the tool axis from the link origin.
    # (M_P = M_O - r_OP x F, Auftrag section 4; here in one axis.)
    tip_b = env._tip_offset_local.reshape(1, 1, 3).repeat(n, 1, 1)
    dz = 0.100  # metres up the tool axis for the second point; own choice
    # THREE loads: x at the tip, x at tip - dz (the lever difference), and
    # y at the tip (the only one that can see an x offset of the reference).
    load_x = torch.zeros(n, 1, 3, device=dev); load_x[:, 0, 0] = float(args_cli.load_n)
    load_y = torch.zeros(n, 1, 3, device=dev); load_y[:, 0, 1] = float(args_cli.load_n)
    points = {"tip": (tip_b.clone(), load_x), "tip_minus_dz": (tip_b.clone(), load_x),
              "tip_y": (tip_b.clone(), load_y)}
    points["tip_minus_dz"][0][:, 0, 2] -= dz
    # The MEASURED reference point (cfg, RT-192a2) as the third candidate.
    ref_off = env._torque_ref_offset_parent
    p2 = {"load_n": float(args_cli.load_n), "dz_m": dz, "points": {},
          "torque_ref_offset_parent_m": _l(ref_off[0]),
          "sign_convention": "incoming wrench = parent reaction = MINUS the load (measured RT-192a first run)"}
    # Where the two centres of mass sit, in the PARENT frame, for the reader:
    # a reference point 0.103 m up the axis has to be SOMETHING's frame.
    p2["com_child_in_parent_m"] = _l(_rot_t(r1["q_parent"], r1["com_w"] - r1["pos_parent"]))
    p2["com_parent_in_parent_m"] = _l(_rot_t(r1["q_parent"], robot.data.body_com_pos_w[:, parent] - r1["pos_parent"]))
    p2["tool_origin_in_parent_m"] = _l(_rot_t(r1["q_parent"], r1["pos_tool"] - r1["pos_parent"]))
    obs_prev, r_prev = obs_rest, r1
    lever_meas = {}
    f_ok_all, wired_all, f_sign = True, True, None
    q0 = robot.data.joint_pos.clone()
    p2["control_mode"] = str(env.cfg.control_mode)
    for pname, (pos_b, load_b) in points.items():
        robot.set_external_force_and_torque(forces=load_b, torques=torch.zeros_like(load_b),
                                            positions=pos_b, body_ids=[tool])
        for _ in range(int(args_cli.settle_steps)):
            obs_load = step()
        r2 = read()
        R_c = insertion_math.axes_from_quat(r2["q_tool"])
        F_w = torch.matmul(R_c, load_b[:, 0].unsqueeze(-1)).squeeze(-1)
        pt_w = r2["pos_tool"] + torch.matmul(R_c, pos_b[:, 0].unsqueeze(-1)).squeeze(-1)
        pred_f = -_rot_t(r2["q_parent"], F_w)
        # RAW deltas: same pose, same gravity, so the difference is the load alone.
        d_tared_f = r2["raw"][:, 0:3] - r1["raw"][:, 0:3]
        d_tared_t = r2["raw"][:, 3:6] - r1["raw"][:, 3:6]
        p2.setdefault("pose_drift_m", {})[pname] = _l(torch.linalg.norm(r2["pos_tool"] - r1["pos_tool"], dim=-1))
        d_ema_t = r2["t_ema"] - r1["t_ema"]
        d_obs_t = obs_load[:, q_lo:q_hi] - obs_rest[:, q_lo:q_hi]
        d_obs_f = obs_load[:, f_lo:f_hi] - obs_rest[:, f_lo:f_hi]
        f_ratio = torch.linalg.norm(d_tared_f, dim=-1) / torch.linalg.norm(pred_f, dim=-1)
        f_cos = _cos(d_tared_f, pred_f)
        # The measured lever, in metres: |d tau| / |F| -- the number the
        # reference point is read from, per application point.
        lever = torch.linalg.norm(d_tared_t, dim=-1) / float(args_cli.load_n)
        lever_meas[pname] = lever
        rec = {"point_local_m": _l(pos_b[:, 0]), "load_local_n": _l(load_b[:, 0]),
               "delta_force_tared_n": _l(d_tared_f), "pred_force_parent_n": _l(pred_f),
               "delta_force_ratio": _l(f_ratio), "delta_force_cos": _l(f_cos),
               "delta_torque_tared_nm": _l(d_tared_t), "delta_torque_ema_nm": _l(d_ema_t),
               "delta_obs_torque_nm": _l(d_obs_t), "delta_obs_force_n": _l(d_obs_f),
               "lever_measured_m": _l(lever), "candidates": {}}
        ref_meas_w = r2["pos_parent"] + torch.matmul(insertion_math.axes_from_quat(r2["q_parent"]),
                                                     ref_off.unsqueeze(-1)).squeeze(-1)
        for cname, ref in (("parent_origin", r2["pos_parent"]), ("child_origin", r2["pos_tool"]),
                           ("measured_ref", ref_meas_w)):
            pred_t = -_rot_t(r2["q_parent"], torch.cross(pt_w - ref, F_w, dim=-1))
            ratio = torch.linalg.norm(d_tared_t, dim=-1) / torch.linalg.norm(pred_t, dim=-1)
            cos = _cos(d_tared_t, pred_t)
            rec["candidates"][cname] = {"pred_torque_nm": _l(pred_t), "ratio": _l(ratio), "cos": _l(cos),
                                        "fits": bool(((ratio >= LOAD_RATIO[0]) & (ratio <= LOAD_RATIO[1])).all())
                                        and bool((cos >= LOAD_COS).all())}
        wired = bool((torch.abs(d_obs_t - d_ema_t) <= 1e-6).all()) and bool(
            (torch.abs(d_obs_f - (r2["f_ema"] - r1["f_ema"])) <= 1e-6).all())
        rec["obs_channel_equals_ema"] = wired
        rec["joint_drift_rad"] = _l(torch.linalg.norm(robot.data.joint_pos - q0, dim=-1))
        rec["force_fits"] = (bool(((f_ratio >= LOAD_RATIO[0]) & (f_ratio <= LOAD_RATIO[1])).all())
                             and bool((f_cos >= LOAD_COS).all()))
        f_ok_all &= rec["force_fits"]
        wired_all &= wired
        p2["points"][pname] = rec
        print(f"[tare] 2 LOAD {args_cli.load_n} N local {[round(v, 1) for v in _l(load_b[0, 0])]} at {pname} {[round(v, 4) for v in _l(pos_b[0, 0])]} m: "
              f"|dF|/|pred| {[round(v, 4) for v in _l(f_ratio)]}, cos {[round(v, 4) for v in _l(f_cos)]}; "
              f"lever measured {[round(v, 4) for v in _l(lever)]} m; "
              f"obs==EMA {wired}")
        for cname, c in rec["candidates"].items():
            print(f"[tare]     ref {cname}: ratio {[round(v, 4) for v in c['ratio']]}, "
                  f"cos {[round(v, 4) for v in c['cos']]} -> {'FITS' if c['fits'] else 'no'}")
    # Clear the load and let the OSC settle before point 3, so no env drifts
    # under a load while the reset is read (a3: env 1 drifted, point 3 FAIL).
    # NOT the documented empty (0, 3) array: the 2.3.2 wrench composer's Warp
    # kernel refuses it ("expects an array with 2 dimension(s)", RT-192a6).
    # Zero forces of the SAME shape at the same point switch the load off.
    robot.set_external_force_and_torque(forces=torch.zeros_like(load_x), torques=torch.zeros_like(load_x),
                                        positions=tip_b, body_ids=[tool])
    for _ in range(int(args_cli.settle_steps)):
        step()
    # THE TWO-POINT READING. Honoured application point: lever(tip) -
    # lever(tip - dz) = dz. Reference point: d = lever(tip) - z_tip, in
    # metres up the tool axis from the link origin (0 = origin, negative =
    # towards the wrist).
    z_tip = float(env._tip_offset_local[2])
    lever_diff = lever_meas["tip"] - lever_meas["tip_minus_dz"]
    d_ref = lever_meas["tip"] - z_tip
    p2["lever_diff_m"] = _l(lever_diff)
    p2["lever_diff_expected_m"] = dz
    p2["application_point_honoured"] = bool((torch.abs(lever_diff - dz) <= 0.03 * dz).all())
    p2["reference_point_d_m"] = _l(d_ref)
    p2["reference_point_reading"] = ("d ~ 0: the link origin (parent/child coincide here); "
                                     "d != 0: the moment is taken about a point d metres up the tool axis "
                                     "-- compare with com_parent_in_parent_m and com_child_in_parent_m above")
    torque_fit = next((c for c in ("parent_origin", "child_origin", "measured_ref")
                       if all(p2["points"][k]["candidates"][c]["fits"] for k in points)), None)
    # The y load reads the x offset of the reference: |d tau_z| / F along -y.
    ty = p2["points"]["tip_y"]["delta_torque_tared_nm"]
    p2["reference_point_x_from_y_load_m"] = [row[2] / float(args_cli.load_n) for row in ty]
    p2["torque_fit"] = torque_fit
    p2["force_fits"] = f_ok_all
    p2["obs_channel_equals_ema"] = wired_all
    # PASS needs: force right at both points, wiring right, application point
    # honoured, AND the moment about the link origin (the env's assumption).
    # PASS: force right at every point, wiring right, application point
    # honoured, AND the moment about the env's (measured) reference point.
    passed["2_load"] = f_ok_all and wired_all and p2["application_point_honoured"] and torque_fit == "measured_ref"
    print(f"[tare] 2 RESULT: force {'fits' if f_ok_all else 'does NOT fit'} at both points; "
          f"lever diff {[round(v, 4) for v in _l(lever_diff)]} m vs dz {dz} -> application point "
          f"{'honoured' if p2['application_point_honoured'] else 'NOT honoured'}; reference point d "
          f"{[round(v, 4) for v in _l(d_ref)]} m up the tool axis; torque fit {torque_fit}; "
          f"obs==EMA {wired_all} -> {'PASS' if passed['2_load'] else 'FAIL'}")
    result["2_load"] = p2

    # ---- 3. PARTIAL RESET ------------------------------------------------
    before = read()
    env._reset_idx(torch.tensor([0], device=dev))
    fresh_raised = bool(env._wrench_fresh[0]) and not bool(env._wrench_fresh[1:].any())
    obs3 = step()
    r3 = read()
    env0_zero = (bool((r3["f_ema"][0] == 0.0).all()) and bool((r3["t_ema"][0] == 0.0).all())
                 and bool((r3["f_tared"][0] == 0.0).all()) and bool((r3["t_tared"][0] == 0.0).all()))
    env0_obs_zero = bool((obs3[0, f_lo:f_hi] == 0.0).all()) and bool((obs3[0, q_lo:q_hi] == 0.0).all())
    valid_ok = (not bool(r3["valid"][0])) and bool(r3["valid"][1:].all())
    others_kept = bool((torch.linalg.norm(r3["t_ema"][1:] - before["t_ema"][1:], dim=-1)
                        <= torch.clamp(REST_REL * torch.linalg.norm(before["t_ema"][1:], dim=-1),
                                       min=REST_ABS_NM)).all())
    obs4 = step()
    r4 = read()
    alpha = float(env.cfg.ft_smoothing_factor)
    restart_ok = (bool(r4["valid"][0]) and bool((torch.abs(r4["f_ema"][0] - alpha * r4["f_tared"][0]) <= 1e-6).all())
                  and bool((torch.abs(r4["t_ema"][0] - alpha * r4["t_tared"][0]) <= 1e-6).all()))
    p3 = {"fresh_flag_raised_only_env0": fresh_raised, "env0_buffers_zero_on_reset_step": env0_zero,
          "env0_obs_force_and_torque_zero": env0_obs_zero,
          "valid_false_env0_true_others": valid_ok, "other_envs_torque_unchanged": others_kept,
          "env0_ema_restarts_from_zero_next_step": restart_ok,
          "env0_next_step_f_ema": _l(r4["f_ema"][0]), "env0_next_step_f_tared": _l(r4["f_tared"][0]),
          "valid_after_reset_step": _l(r3["valid"]), "valid_next_step": _l(r4["valid"])}
    passed["3_reset"] = fresh_raised and env0_zero and env0_obs_zero and valid_ok and others_kept and restart_ok
    print(f"[tare] 3 PARTIAL RESET: flag {fresh_raised}, env0 zero {env0_zero}, obs zero {env0_obs_zero}, "
          f"valid {valid_ok}, others kept {others_kept}, restart {restart_ok} -> "
          f"{'PASS' if passed['3_reset'] else 'FAIL'}")
    result["3_reset"] = p3

    # ---- 4. DOUBLE READ --------------------------------------------------
    a = (env._force_smooth.clone(), env._torque_smooth.clone(),
         env._force_tared_raw.clone(), env._torque_tared_raw.clone(), env._wrench_valid.clone())
    env._get_observations()
    env._get_observations()
    b = (env._force_smooth, env._torque_smooth, env._force_tared_raw, env._torque_tared_raw, env._wrench_valid)
    same = [bool(torch.equal(x, y)) for x, y in zip(a, b)]
    p4 = {"buffers_byte_identical": same,
          "names": ["_force_smooth", "_torque_smooth", "_force_tared_raw", "_torque_tared_raw", "_wrench_valid"]}
    passed["4_double_read"] = all(same)
    print(f"[tare] 4 DOUBLE READ: {dict(zip(p4['names'], same))} -> "
          f"{'PASS' if passed['4_double_read'] else 'FAIL'}")
    result["4_double_read"] = p4

    result["passed"] = passed
    ok = all(passed.values())
    result["verdict"] = "TARE_CHECK_PASS" if ok else "TARE_CHECK_FAIL"
    print(f"[tare] VERDICT: {result['verdict']} {passed}")
    return finish(result, env, ok=ok)


def finish(result: dict, env=None, ok: bool = False) -> int:
    default_name = "wrench_tare_check.json" if args_cli.tare_check else "wrench_bodies.json"
    out = pathlib.Path(args_cli.json) if args_cli.json else (
        pathlib.Path(__file__).resolve().parent / default_name)
    out.write_text(json.dumps(result, indent=2))
    print(f"[wrench] metrics written to {out}")
    if env is not None:
        try:
            env.close()
        except Exception:  # noqa: BLE001 - closing must never mask the result
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    _code = 1
    try:
        _code = main_tare_check() if args_cli.tare_check else main_bodies()
    except BaseException:  # noqa: BLE001 - D-081: a silent crash reported exit 0 before
        traceback.print_exc()
        _code = 1
    finally:
        exit_with(simulation_app, _code)

"""Rank the training runs and name the best policy's checkpoint.

Reads every run under ``logs/rsl_rl/<experiment>/`` and reports each one's
recent success rate, the configuration it was trained under, and the
checkpoint file to load. Needs no Isaac, no PyTorch and no running
simulation: it reads the ``demo_metrics.json`` each run writes into its own
log directory.

Ranking uses **success_rate_recent** (the trailing window the environment
keeps), not the cumulative rate. The cumulative figure averages in the
untrained start of the run, so a policy that learned late looks worse than
it is -- the wrong basis for picking a checkpoint.

    python scripts/compare_runs.py
    python scripts/compare_runs.py --experiment demo_insertion --top 5

Caveat worth stating when these numbers are used: the runs are ranked on
their own training distribution. A run trained with little start
randomisation can top this table and still fail under wider starts, so
compare like with like, and read reset_joint_noise_rad in the table before
concluding one policy is better than another.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def read_config_yaml(run_dir: pathlib.Path) -> dict:
    """Configuration a run was trained under, from the params/env.yaml that
    train.py dumps. This is how runs predating the per-run metrics file can
    still be told apart -- without it they are indistinguishable timestamps.
    """
    path = run_dir / "params" / "env.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    try:
        cfg = yaml.unsafe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001 - a malformed dump must not stop the ranking
        return {}
    if not isinstance(cfg, dict):
        return {}
    held = cfg.get("held_asset") or {}
    diameter = held.get("diameter") if isinstance(held, dict) else None
    fixed = cfg.get("fixed_asset") or {}
    bore = fixed.get("diameter") if isinstance(fixed, dict) else None
    out = {
        "reset_joint_noise_rad": cfg.get("reset_joint_noise"),
        "episode_length_s": cfg.get("episode_length_s"),
        "peg_diameter_m": diameter,
    }
    if isinstance(bore, (int, float)) and isinstance(diameter, (int, float)):
        out["radial_clearance_mm"] = (bore - diameter) / 2.0 * 1000.0
    scene = cfg.get("scene") or {}
    if isinstance(scene, dict):
        out["num_envs"] = scene.get("num_envs")
    return {k: v for k, v in out.items() if v is not None}


def read_tensorboard_success(run_dir: pathlib.Path) -> dict:
    """Last logged success rate from the run's TensorBoard events.

    Recovers runs written before the per-run metrics file existed, whose
    numbers went to one shared path and overwrote each other. Needs the
    tensorboard package (present where training runs, absent on a bare dev
    PC); returns {} rather than failing if it is missing.

    Tag names confirmed on the training machine (2026-07-27): a pre-fix run
    exposes ``Episode/success_rate`` and ``Episode/mean_final_depth_mm``,
    which are the first and second candidates below. Runs written after the
    metrics fix publish ``Episode/success_rate`` as the trailing window plus
    ``Episode/success_rate_cumulative`` and ``Episode/mean_max_depth_mm``;
    those carry a metrics file, so this path is not reached for them.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError:
        return {}
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        return {}
    try:
        ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
    except Exception:  # noqa: BLE001
        return {}

    def last(*candidates: str):
        for tag in candidates:
            if tag in available:
                try:
                    points = ea.Scalars(tag)
                except Exception:  # noqa: BLE001
                    continue
                if points:
                    return points[-1].value, len(points)
        return None, 0

    rate, n = last("Episode/success_rate", "Episode/success_rate_cumulative", "Train/success_rate")
    if rate is None:
        return {}
    depth, _ = last("Episode/mean_max_depth_mm", "Episode/mean_final_depth_mm")
    out = {"success_rate_recent": rate, "success_rate_cumulative": rate, "_from_tensorboard": True,
           "recent_window_episodes": 0, "episodes": n}
    if depth is not None:
        out["mean_max_depth_mm"] = depth
    return out


def find_checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """Checkpoints in ascending iteration order (model_0.pt, model_50.pt, ...)."""

    def iteration(p: pathlib.Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    return sorted(run_dir.glob("model_*.pt"), key=iteration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="demo_insertion", help="Experiment folder under logs/rsl_rl/.")
    parser.add_argument("--logs", default="logs", help="Root logs directory.")
    parser.add_argument("--top", type=int, default=0, help="Show only the best N runs (0 = all).")
    args = parser.parse_args()

    root = pathlib.Path(args.logs) / "rsl_rl" / args.experiment
    if not root.is_dir():
        print(f"No such experiment directory: {root.resolve()}")
        print("Run training first, or pass --experiment / --logs.")
        return 1

    rows = []
    missing = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_file = run_dir / "demo_metrics.json"
        checkpoints = find_checkpoints(run_dir)
        m: dict = {}
        if metrics_file.is_file():
            try:
                # utf-8-sig: tolerate a byte-order mark, which some Windows
                # editors add and json.loads otherwise chokes on.
                m = json.loads(metrics_file.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                missing.append((run_dir.name, f"unreadable: {exc}"))
                continue
        else:
            # Runs from before the per-run metrics file: recover what can be
            # recovered rather than dropping real training time.
            m = read_tensorboard_success(run_dir)
            if not m:
                missing.append((run_dir.name, len(checkpoints)))
                continue
        # The config dump fills in whatever the metrics file does not carry,
        # which for recovered runs is everything.
        for key, value in read_config_yaml(run_dir).items():
            m.setdefault(key, value)
        rows.append(
            {
                "run": run_dir.name,
                "recent": m.get("success_rate_recent", 0.0),
                "cumulative": m.get("success_rate_cumulative", 0.0),
                "episodes": m.get("episodes", 0),
                "window": m.get("recent_window_episodes", 0),
                "depth_mm": m.get("mean_max_depth_mm", 0.0),
                "to_threshold": m.get("episodes_to_threshold"),
                "clearance_mm": m.get("radial_clearance_mm"),
                "noise_rad": m.get("reset_joint_noise_rad"),
                "envs": m.get("num_envs"),
                "checkpoint": checkpoints[-1] if checkpoints else None,
                "n_checkpoints": len(checkpoints),
                "recovered": bool(m.get("_from_tensorboard")),
            }
        )

    if not rows:
        print(f"No run in {root.resolve()} has a demo_metrics.json yet.")
        if missing:
            print("Runs found without metrics (too short, or an older code version):")
            for name, info in missing:
                print(f"  {name}  ({info} checkpoints)" if isinstance(info, int) else f"  {name}  ({info})")
        return 1

    # Rate first, then sample efficiency. Once several configurations all
    # finish at 100 % the rate stops separating them, and the episode count
    # at which each first held the threshold is what does. Runs that never
    # reached it sort last within their rate.
    def sort_key(r):
        reached = r["to_threshold"] if isinstance(r["to_threshold"], int) else float("inf")
        return (-round(r["recent"], 4), reached)

    rows.sort(key=sort_key)
    shown = rows[: args.top] if args.top > 0 else rows

    def fmt(value, spec: str, dash: str = "  -  ") -> str:
        return format(value, spec) if isinstance(value, (int, float)) else dash

    print(f"\n{len(rows)} run(s) in {root.resolve()}, ranked by success rate, then by how fast\n")
    header = (f"{'run':<34} {'recent':>7} {'to 90%':>9} {'cumul':>7} {'depth':>7} "
              f"{'clear':>6} {'noise':>6} {'envs':>5} {'eps':>7}")
    print(header)
    print("-" * len(header))
    for r in shown:
        mark = " *" if r["recovered"] else ""
        reached = f"{r['to_threshold']:>9d}" if isinstance(r["to_threshold"], int) else f"{'never':>9}"
        print(
            f"{r['run']:<34} {r['recent']:>6.1%} {reached} {r['cumulative']:>6.1%} "
            f"{fmt(r['depth_mm'], '>6.1f')} {fmt(r['clearance_mm'], '>5.2f')} "
            f"{fmt(r['noise_rad'], '>5.3f')} {fmt(r['envs'], '>5d')} {r['episodes']:>7d}{mark}"
        )
    print("\n(recent = trailing window of episodes; 'to 90%' = episodes needed before the")
    print(" recent rate first held the threshold, i.e. sample efficiency -- the figure that")
    print(" separates configurations once they all finish at 100 %; depth in mm; clear =")
    print(" radial clearance in mm; noise = reset joint noise in rad.)")
    if any(r["recovered"] for r in rows):
        print("\n* Recovered from TensorBoard, not from a metrics file: these runs predate")
        print("  the per-run metrics and report the CUMULATIVE rate in both columns, which")
        print("  understates a policy that learned late. Treat as a lower bound, and do not")
        print("  rank them head-to-head against runs that have a real trailing window.")

    best = rows[0]
    print(f"\nBest run: {best['run']}")
    print(f"  recent success rate : {best['recent']:.1%} over {best['window']} episodes")
    if best["checkpoint"] is not None:
        print(f"  latest checkpoint   : {best['checkpoint']}")
        # play.py passes --checkpoint through retrieve_file_path, i.e. it is a
        # PATH, not a filename inside --load_run (which is how train.py reads
        # it). Printing the bare filename here produced a command that looked
        # right and failed.
        print("\nWatch it run (windowed, real time):")
        print(
            f"  C:\\Isaaclab\\isaaclab.bat -p .\\scripts\\rsl_rl\\play.py "
            f"--task=Template-Proxytask-Direct-v0 --num_envs 16 --real-time "
            f"--checkpoint {best['checkpoint']}"
        )
        if isinstance(best["clearance_mm"], (int, float)):
            peg_mm = round(30.0 - 2 * best["clearance_mm"])
            print(
                f"\n  This policy was trained on the {peg_mm} mm peg. Set the rung to match\n"
                f"  before replaying it, or the geometry will not be the one it learned:\n"
                f"    $env:PROXYTASK_PEG_DIAMETER_MM = \"{peg_mm}\""
            )
    else:
        print("  no checkpoint written yet (run shorter than save_interval)")

    if missing:
        print(f"\n{len(missing)} run(s) without metrics, not ranked:")
        for name, info in missing:
            print(f"  {name}  ({info} checkpoints)" if isinstance(info, int) else f"  {name}  ({info})")

    # Same-configuration groups are the only fair head-to-head comparisons.
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["clearance_mm"], r["noise_rad"]), []).append(r)
    comparable = {k: v for k, v in groups.items() if len(v) > 1}
    if comparable:
        print("\nRuns sharing a configuration (fair comparisons):")
        for (clear, noise), members in comparable.items():
            head = f"  clearance {clear} mm, noise {noise} rad:"
            print(head)
            for r in members:
                print(f"    {r['recent']:>6.1%}  {r['run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Run curves over TOTAL env steps from a ``RT-<N>_scalars.csv``, PNG + PDF.

WHAT IT DRAWS
-------------
One figure per curve, each as ``<out-dir>/RT-<N>_<curve>.png`` AND ``.pdf``
(same figure, two ``savefig`` calls; the PDF is the thesis copy):

    reward       mean episode reward
    success      trailing-window success rate, in %
    depth        mean maximum insertion depth, mm
    force_abort  force-abort rate, in %

An optional reference run (``--ref``, ``--ref-run``) is drawn on the same
axes in a second colour; the legend names the two run IDs.

THE X AXIS (D-195 CURVE AXIS): total env steps, all envs summed,
``tools/run_axis.py``. Each run gets ITS OWN factor from its own
``params/agent.yaml`` and ``params/env.yaml`` (``--agent-yaml``/``--env-yaml``,
``--ref-agent-yaml``/``--ref-env-yaml``), so a 16-step and a 128-step run
share one axis. Wall time is one console line per run, never a figure.
``--smooth`` is a window in logged points, not in steps.

WHERE THE TAG NAMES COME FROM
-----------------------------
The CSV is the wide table of ``scripts/export_tb_scalars.py`` (header
``step,wall_time,<tag>,...``). rsl_rl versions name their tags differently
and the runner source is not on the laptop, so each curve has a CANDIDATE
list and the first tag present wins. A curve none of whose candidates is in
the file is SKIPPED with the file's tag list printed -- extend the list, or
pass ``--reward-tag`` to pick the reward series by hand.

Offline, no Isaac: ``csv`` + ``matplotlib``. Style follows
``plot_start_height_bins.py`` (one hue per series, German axis labels with
units, ``--no-title`` for thesis figures whose caption names them).

    python scripts/plot_run_curves.py --scalars docs/figures/RT-235s1_scalars.csv --run RT-235s1 \
        --agent-yaml rt_logs/RT-235s1/RT-235s1_agent.yaml --env-yaml rt_logs/RT-235s1/RT-235s1_env.yaml \
        --out-dir docs/figures
    python scripts/plot_run_curves.py --self-test

``--self-test`` draws SYNTHETIC curves into a temp dir, checks the tag
resolver (first candidate present, skip on miss), the header guard, the
two output files, each run's own x grid through ``main`` and the named
mutation "reference drawn with the main run's factor"; it never writes
under ``docs/``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
import tempfile

_spec = importlib.util.spec_from_file_location("run_axis", pathlib.Path(__file__).resolve().parent / "tools" / "run_axis.py")
run_axis = importlib.util.module_from_spec(_spec)
sys.modules["run_axis"] = run_axis
_spec.loader.exec_module(run_axis)

HEADER_FIXED = ("step", "wall_time")

RUN = "#3B6FD4"
REF = "#D97706"
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"

CURVES: dict[str, tuple[tuple[str, ...], str, float, tuple[float, float] | None]] = {
    "reward": (("Train/mean_reward", "Episode_Reward/total"), "Mittlerer Episoden-Reward [–]", 1.0, None),
    "success": (("Episode/success_rate", "success_rate"), "Erfolgsrate [%]", 100.0, (0.0, 100.0)),
    "depth": (("Episode/mean_max_depth_mm", "mean_max_depth_mm"), "Mittlere maximale Fügetiefe [mm]", 1.0, None),
    "force_abort": (("Episode/force_abort_rate", "force_abort_rate"), "Kraftabbruch-Rate [%]", 100.0, (0.0, 100.0)),
}
TITLES = {"reward": "Reward", "success": "Erfolgsrate", "depth": "Fügetiefe", "force_abort": "Kraftabbruch-Rate"}


def read_csv(path: pathlib.Path) -> tuple[list[str], list[int], list[float], dict[str, list[float | None]]]:
    """(tags, steps, wall times, {tag: values aligned to steps, None where empty})."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if header is None or tuple(header[:2]) != HEADER_FIXED:
            raise ValueError(f"{path}: header must start with {','.join(HEADER_FIXED)}; got {header}")
        tags = header[2:]
        steps: list[int] = []
        walls: list[float] = []
        series: dict[str, list[float | None]] = {t: [] for t in tags}
        for cells in r:
            if not cells:
                continue
            steps.append(int(cells[0]))
            walls.append(float(cells[1]))
            for tag, cell in zip(tags, cells[2:]):
                series[tag].append(float(cell) if cell != "" else None)
    return tags, steps, walls, series


def load_run(scalars: pathlib.Path, label: str, factor: tuple[int, int]
             ) -> tuple[list[str], list[int], dict[str, list[float | None]]]:
    """(tags, x in total env steps, series) of one run, with ITS OWN factor ``(num_steps_per_env, num_envs)``; prints the run's wall-time line."""
    tags, steps, walls, series = read_csv(scalars)
    it = [i for i in range(len(steps)) if any(series[t][i] is not None for t in tags if not t.endswith("/time"))]
    print(f"  {label}: " + run_axis.wall_time_line([steps[i] for i in it], [walls[i] for i in it]))
    return tags, [run_axis.env_steps(s, *factor) for s in steps], series


def resolve(tags: list[str], candidates: tuple[str, ...]) -> str | None:
    """The first candidate that is a tag of the file; None if none is."""
    for c in candidates:
        if c in tags:
            return c
    return None


def smooth(values: list[float | None], window: int) -> list[float | None]:
    """Trailing moving average over ``window`` samples; None passes through as a gap."""
    if window <= 1:
        return list(values)
    out: list[float | None] = []
    buf: list[float] = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def draw_curve(curve: str, runs: list[tuple[str, list[int], list[float | None]]], out_base: pathlib.Path,
               title: bool = True, tag_note: str = "") -> list[pathlib.Path]:
    """Draw one curve for one or two runs (x already in total env steps); returns the two files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ylabel, scale, ylim = CURVES[curve]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for (label, x, values), colour in zip(runs, (RUN, REF)):
        xs = [s for s, v in zip(x, values) if v is not None]
        ys = [v * scale for v in values if v is not None]
        ax.plot(xs, ys, color=colour, linewidth=1.4, label=label, zorder=3)
    ax.set_xlabel("Env-Schritte (alle Envs)", color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    if len(runs) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    if title:
        text = f"{runs[0][0]}: {TITLES[curve]} über Env-Schritte"
        if tag_note:
            text += f"  ({tag_note})"
        ax.set_title(text, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in (".png", ".pdf"):
        p = out_base.with_suffix(ext)
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written


def plot_all(scalars: pathlib.Path, run: str, factor: tuple[int, int], out_dir: pathlib.Path, which: list[str],
             ref: pathlib.Path | None, ref_run: str, ref_factor: tuple[int, int] | None, reward_tag: str | None,
             window: int, title: bool) -> int:
    """Draw the requested curves; returns how many were drawn."""
    tags, steps, series = load_run(scalars, run, factor)
    ref_data = load_run(ref, ref_run, ref_factor) if ref is not None else None
    drawn = 0
    for curve in which:
        candidates = CURVES[curve][0]
        if curve == "reward" and reward_tag:
            candidates = (reward_tag,)
        tag = resolve(tags, candidates)
        if tag is None:
            print(f"SKIP {curve}: none of {list(candidates)} in {scalars.name}; tags present: {tags}")
            continue
        runs = [(run, steps, smooth(series[tag], window))]
        if ref_data is not None:
            r_tags, r_steps, r_series = ref_data
            r_tag = resolve(r_tags, candidates)
            if r_tag is None:
                print(f"  reference {ref_run}: none of {list(candidates)} in {ref.name}; drawn without reference")
            else:
                runs.append((ref_run, r_steps, smooth(r_series[r_tag], window)))
        files = draw_curve(curve, runs, out_dir / f"{run}_{curve}", title=title, tag_note=tag)
        print(f"  {curve:<12} <- {tag:<32} " + ", ".join(str(p) for p in files))
        drawn += 1
    return drawn


# D-195 axis self-test: main() on a synthetic yaml pair
AXIS_CHECKS = (
    "main run: first x = (0 + 1) * 128 * 4 = 512",
    "main run: every x = (step + 1) * 128 * 4",
    "reference (16 steps per env) lands on its own grid (step + 1) * 16 * 4",
    "wall-time line counts iteration rows only (not the */time rows)",
    "--ref without --ref-agent-yaml / --ref-env-yaml is refused",
)
AXIS_MUTATIONS = (
    ("reference-drawn-with-the-main-factor", "load_run(ref, ref_run, ref_factor)", "load_run(ref, ref_run, factor)",
     (AXIS_CHECKS[2],)),
)
_SELF_TEST_MARKER = "# D-195 axis self-test: main() on a synthetic yaml pair"


def _axis_checks(text: str, root: pathlib.Path) -> dict[str, bool]:
    """Run ``main`` of the source ``text`` on two synthetic runs; ``draw_curve`` is replaced by a recorder, so the x values reach the checks undrawn."""
    import contextlib
    import io

    def pair(name: str, n: int, envs: int) -> list[str]:
        a, e = root / f"{name}_agent.yaml", root / f"{name}_env.yaml"
        a.write_text(f"seed: 1\nnum_steps_per_env: {n}\n", encoding="utf-8")
        e.write_text(f"seed: 1\nscene:\n  num_envs: {envs}\n  env_spacing: 3.0\n", encoding="utf-8")
        return [str(a), str(e)]

    main_csv, ref_csv = root / "axis_main.csv", root / "axis_ref.csv"
    main_csv.write_text("step,wall_time,Train/mean_reward\n"
                        + "".join(f"{i},{1000.0 + 60 * i},{i}\n" for i in range(30)), encoding="utf-8")
    ref_csv.write_text("step,wall_time,Train/mean_reward,Train/mean_reward/time\n"
                       + "".join(f"{i},{1000.0 + 360 * i},{i},\n" for i in range(10))
                       + "50,99999.0,,1\n60,99999.0,,2\n", encoding="utf-8")
    a, e = pair("axis_main", 128, 4)
    ra, re_ = pair("axis_ref", 16, 4)
    base = ["--scalars", str(main_csv), "--run", "A", "--agent-yaml", a, "--env-yaml", e,
            "--ref", str(ref_csv), "--ref-run", "B", "--which", "reward", "--out-dir", str(root / "axis")]
    ns = {"__name__": "plot_run_curves_under_test", "__file__": __file__}
    exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
    drawn: list = []
    ns["draw_curve"] = lambda curve, runs, out_base, title=True, tag_note="": drawn.append(runs) or []
    out = io.StringIO()
    res = dict.fromkeys(AXIS_CHECKS, False)
    try:
        with contextlib.redirect_stdout(out):
            ns["main"](base + ["--ref-agent-yaml", ra, "--ref-env-yaml", re_])
        (_, xa, _), (_, xb, _) = drawn[0]
        res[AXIS_CHECKS[0]] = xa[0] == 512
        res[AXIS_CHECKS[1]] = xa == [(s + 1) * 512 for s in range(30)]
        res[AXIS_CHECKS[2]] = xb == [(s + 1) * 64 for s in list(range(10)) + [50, 60]]
        res[AXIS_CHECKS[3]] = ("B: wall time 0.90 h over 10 iterations (steps 0..9, last minus first wall_time)"
                               in out.getvalue())
    except BaseException:   # noqa: BLE001 - every check stays False
        pass
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            ns["main"](base)
    except SystemExit as exc:
        res[AXIS_CHECKS[4]] = exc.code == 2
    except BaseException:   # noqa: BLE001
        pass
    return res


def _self_test() -> int:
    rc = 0

    def check(name: str, ok: bool) -> None:
        nonlocal rc
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    check("resolver takes the FIRST present candidate", resolve(["b", "a"], ("a", "b")) == "a")
    check("resolver falls through to a later candidate", resolve(["b"], ("a", "b")) == "b")
    check("resolver returns None on a miss", resolve(["c"], ("a", "b")) is None)
    check("smooth window 1 is the identity", smooth([1.0, None, 3.0], 1) == [1.0, None, 3.0])
    check("smooth window 2 averages and keeps gaps", smooth([1.0, 3.0, None, 5.0], 2) == [1.0, 2.0, None, 4.0])

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        csv_path = root / "synthetic_scalars.csv"
        lines = ["step,wall_time,Train/mean_reward,Episode/success_rate"]
        for i in range(30):
            lines.append(f"{i},{1.7e9 + i:.3f},{10.0 * i:.7g},{i / 30.0:.7g}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tags, steps, walls, series = read_csv(csv_path)
        check("read_csv returns tags, 30 steps, 30 wall times, aligned series",
              tags == ["Train/mean_reward", "Episode/success_rate"]
              and len(steps) == 30 and len(walls) == 30 and len(series[tags[0]]) == 30)

        bad = root / "bad.csv"
        bad.write_text("iteration,reward\n1,2\n", encoding="utf-8")
        try:
            read_csv(bad)
            check("a foreign header is refused", False)
        except ValueError:
            check("a foreign header is refused", True)

        out_dir = root / "figs"
        n = plot_all(csv_path, "SYNTHETIC", (128, 4), out_dir, ["reward", "success", "depth"], None, "", None,
                     None, 1, True)
        check("two curves drawn, the third (depth) skipped without exception", n == 2)
        for curve in ("reward", "success"):
            for ext in (".png", ".pdf"):
                p = out_dir / f"SYNTHETIC_{curve}{ext}"
                check(f"{p.name} written (> 1000 bytes)", p.is_file() and p.stat().st_size > 1000)
        check("skipped curve leaves no file", not (out_dir / "SYNTHETIC_depth.png").exists())

        n = plot_all(csv_path, "SYNTHETIC", (128, 4), out_dir, ["reward"], csv_path, "REF", (16, 4), None, 3, False)
        check("reference overlay + smoothing + --no-title draws", n == 1)

        src = pathlib.Path(__file__).read_text(encoding="utf-8")
        for name, ok in _axis_checks(src, root).items():
            check(name, ok)
        cut = src.index(_SELF_TEST_MARKER)
        head, tail = src[:cut], src[cut:]
        for mname, old, new, want in AXIS_MUTATIONS:
            if head.count(old) != 1:
                check(f"mutation {mname}: anchor matches {head.count(old)} times", False)
                continue
            got = _axis_checks(head.replace(old, new) + tail, root)
            flipped = {k for k, v in got.items() if not v}
            check(f"mutation {mname} flips exactly {sorted(want)}"
                  + ("" if flipped == set(want) else f" (flipped {sorted(flipped)})"), flipped == set(want))

    print("SELF-TEST PASSED" if rc == 0 else "SELF-TEST FAILED")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=pathlib.Path, help="docs/figures/RT-<N>_scalars.csv")
    ap.add_argument("--run", default="", help="run label, e.g. RT-156 (also the file-name prefix)")
    ap.add_argument("--agent-yaml", type=pathlib.Path, default=None,
                    help="the run's params/agent.yaml (num_steps_per_env)")
    ap.add_argument("--env-yaml", type=pathlib.Path, default=None, help="the run's params/env.yaml (scene.num_envs)")
    ap.add_argument("--ref", type=pathlib.Path, default=None, help="reference run CSV to overlay")
    ap.add_argument("--ref-run", default="Referenz", help="label of the reference run, e.g. RT-138")
    ap.add_argument("--ref-agent-yaml", type=pathlib.Path, default=None, help="the reference run's params/agent.yaml")
    ap.add_argument("--ref-env-yaml", type=pathlib.Path, default=None, help="the reference run's params/env.yaml")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    ap.add_argument("--which", default="reward,success,depth,force_abort",
                    help="comma list of curves: " + ",".join(CURVES))
    ap.add_argument("--reward-tag", default=None, help="tag to use for the reward curve instead of the candidates")
    ap.add_argument("--smooth", type=int, default=1,
                    help="trailing moving-average window in logged points, not in steps (1 = off)")
    ap.add_argument("--no-title", action="store_true", help="omit the in-figure title (thesis: the caption names it)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    if a.scalars is None or not a.run or a.agent_yaml is None or a.env_yaml is None:
        ap.error("--scalars, --run, --agent-yaml and --env-yaml are required (or --self-test)")
    if a.ref is not None and (a.ref_agent_yaml is None or a.ref_env_yaml is None):
        ap.error("--ref needs --ref-agent-yaml and --ref-env-yaml: each run is drawn with its own factor (D-195)")
    which = [w.strip() for w in a.which.split(",") if w.strip()]
    unknown = [w for w in which if w not in CURVES]
    if unknown:
        ap.error(f"unknown curve(s) {unknown}; known: {list(CURVES)}")
    factor = run_axis.read_factor(a.agent_yaml, a.env_yaml)
    ref_factor = run_axis.read_factor(a.ref_agent_yaml, a.ref_env_yaml) if a.ref is not None else None
    n = plot_all(a.scalars, a.run, factor, a.out_dir, which, a.ref, a.ref_run, ref_factor, a.reward_tag, a.smooth,
                 not a.no_title)
    print(f"drawn {n} of {len(which)} curves")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""One home for the training-curve x axis: TOTAL env steps (D-195 CURVE AXIS).

THE RULE. x at logged iteration ``it`` = ``(it + 1) * num_steps_per_env *
num_envs``. rsl_rl 3.0.1 ``runners/on_policy_runner.py`` (``v3.0.1`` tag):
``learn()`` calls ``self.log(locals())`` AFTER the rollout and the update of
iteration ``it``, and ``log()`` writes every scalar at step ``it``. So the
row ``step = it`` of ``scalars.csv`` stands after ``it + 1`` rollouts of
``num_steps_per_env`` steps in each of ``num_envs`` envs.

WHERE THE TWO FACTORS COME FROM: each run's own ``params/agent.yaml`` and
``params/env.yaml``, never typed.

* ``num_steps_per_env``: exactly one top-level key of ``agent.yaml``.
* ``num_envs``: exactly one ``num_envs:`` in the whole of ``env.yaml``, and
  it sits directly under the top-level ``scene:`` block. A second one
  anywhere, or the only one under another block, is refused: the reader
  never guesses which of two counts the run used.

Missing or twice -> ``SystemExit`` naming the file and the key (fail fast,
as ``plot_autodr_run.need``). Own construction: no existing reader takes a
nested key (``eval_figures._env_value`` reads top-level keys only).

WALL TIME gets one line of text, never a figure (D-195): ``wall_time_line``,
last minus first ``wall_time`` of the iteration rows the caller passes.

OWN LIMIT. A RESUMED run counts from ``start_iter``, not 0; the rule above
holds for a fresh run only. The reader cannot see a resume and refuses
nothing about it. This conversion assumes fresh runs.

RL_GAMES (D-200). The factor is
``(horizon_length, num_envs)``: exactly one ``horizon_length:`` in
``agent.yaml``, directly under ``params:`` -> ``config:``; ``num_envs`` by the
rule above. rl_games writes its tags on THREE step axes (measured in
``rt_logs/RT-240c/RT-240c_scalars.csv``; D-200), with F = horizon_length x num_envs:

* ``Episode/*``, ``*/iter``: step = epoch, from 1 -> x = step * F
* ``info/*`` (not ``info/cval_lr``), ``losses/a_loss|c_loss|entropy|bounds_loss``,
  ``performance/*``, ``*/step``: step = (epoch - 1) * F -> x = step + F
* ``losses/cval_loss``, ``info/cval_lr``: step = epoch * F -> x = step
* ``*/time``: seconds -> no curve; the largest is the wall time

x after epoch e is e * F, the same count as rsl_rl's (it + 1) * F with
it = e - 1, so both libraries share one axis. The three families can share a
``step`` value (epoch 87 and 87 s land in one CSV row): read per TAG, never
per row. ``check_rl_games_epochs`` refuses a CSV whose ``info/epochs`` rows
disagree with F.

``read_policy_dt``: the policy step in s = ``sim.dt`` x the decimation the
env RUNS with (the ``dauer`` figure of ``plot_autodr_run.py``), same
exactly-one rule. The dumped ``env.yaml`` holds the fields BEFORE
``resolve_control_mode`` (``insertion_env_cfg.py``) writes ``osc_decimation``
into ``decimation``: an OSC run's ``decimation:`` is the joint-PD 60 Hz
(D-024), its policy runs at sim.dt x ``osc_decimation`` (15 Hz). So the key
follows ``control_mode:``.

Pure and stdlib, no yaml parser: the keys are read line by line.

    python scripts/tools/run_axis.py --self-test
"""

from __future__ import annotations

import pathlib
import re
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_]\w*):(?P<rest>.*)$")


def _lines(path) -> list[str]:
    return pathlib.Path(path).read_text(encoding="utf-8").splitlines()


def _int_value(path, key: str, no: int, rest: str) -> int:
    try:
        return int(rest.strip())
    except ValueError:
        raise SystemExit(f"{path}:{no}: '{key}:' holds {rest.strip()!r}, not an integer") from None


def read_factor(agent_yaml, env_yaml) -> tuple[int, int]:
    """``(num_steps_per_env, num_envs)`` of one run, read from its two
    ``params`` files. Refuses (``SystemExit``) on a missing or doubled key."""
    steps = [(no, m.group("rest")) for no, line in enumerate(_lines(agent_yaml), 1)
             if (m := _KEY.match(line)) and not m.group("indent") and m.group("key") == "num_steps_per_env"]
    if len(steps) != 1:
        raise SystemExit(f"{agent_yaml}: expected exactly one top-level 'num_steps_per_env:', found {len(steps)}")
    return (_int_value(agent_yaml, "num_steps_per_env", *steps[0]), _scene_num_envs(env_yaml))


def _key_paths(path, key: str) -> list[tuple[int, list[str], str]]:
    """Every ``key:`` line of a yaml file as (line number, parent keys, value)."""
    hits, stack = [], []   # stack: (indent, key) of the open blocks
    for no, line in enumerate(_lines(path), 1):
        m = _KEY.match(line)
        if m is None:
            continue
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if m.group("key") == key:
            hits.append((no, [k for _, k in stack], m.group("rest")))
        stack.append((indent, m.group("key")))
    return hits


def read_factor_rl_games(agent_yaml, env_yaml) -> tuple[int, int]:
    """``(horizon_length, num_envs)`` of one rl_games run. Refuses unless
    exactly one ``horizon_length:`` exists and it sits under params -> config."""
    hits = _key_paths(agent_yaml, "horizon_length")
    if len(hits) != 1 or hits[0][1] != ["params", "config"]:
        where = ", ".join(f"line {h[0]} under {'.'.join(h[1]) or 'top level'}" for h in hits) or "none"
        raise SystemExit(f"{agent_yaml}: expected exactly one 'horizon_length:', directly under "
                         f"'params:' -> 'config:'; found {len(hits)} ({where})")
    return _int_value(agent_yaml, "horizon_length", hits[0][0], hits[0][2]), _scene_num_envs(env_yaml)


def read_run_factor(agent_yaml, env_yaml) -> tuple[str, tuple[int, int]]:
    """``("rsl_rl" | "rl_games", factor)``: the library from which key the
    ``agent.yaml`` carries; both or neither is refused."""
    rsl = any((m := _KEY.match(line)) and not m.group("indent") and m.group("key") == "num_steps_per_env"
              for line in _lines(agent_yaml))
    rlg = bool(_key_paths(agent_yaml, "horizon_length"))
    if rsl == rlg:
        raise SystemExit(f"{agent_yaml}: cannot tell the library (top-level 'num_steps_per_env:' "
                         f"{'and' if rsl else 'nor'} 'horizon_length:')")
    return ("rsl_rl", read_factor(agent_yaml, env_yaml)) if rsl else \
        ("rl_games", read_factor_rl_games(agent_yaml, env_yaml))


_CONTROL_MODES = ("osc", "joint_pd")   # insertion_env_cfg.CONTROL_MODES


def _one_top_level(path, key: str) -> tuple[int, str]:
    """(line number, value) of the one top-level ``key:``; else refused."""
    hits = _key_paths(path, key)
    if len(hits) != 1 or hits[0][1] != []:
        raise SystemExit(f"{path}: expected exactly one top-level '{key}:'; "
                         f"found {[(h[0], h[1]) for h in hits]}")
    return hits[0][0], hits[0][2]


def read_policy_dt(env_yaml) -> float:
    """Policy step in s: ``sim.dt`` x the decimation ``resolve_control_mode``
    runs with: ``osc`` -> ``osc_decimation``, ``joint_pd`` -> ``decimation``.
    Exactly one ``dt:`` in the file, directly under top-level ``sim:``;
    exactly one top-level ``control_mode:`` and one of the key it names."""
    dts = _key_paths(env_yaml, "dt")
    if len(dts) != 1 or dts[0][1] != ["sim"]:
        raise SystemExit(f"{env_yaml}: expected exactly one 'dt:' directly under 'sim:'; "
                         f"found {[(h[0], h[1]) for h in dts]}")
    try:
        dt = float(dts[0][2].strip())
    except ValueError:
        raise SystemExit(f"{env_yaml}:{dts[0][0]}: 'dt:' holds {dts[0][2].strip()!r}, not a number") from None
    mode = _one_top_level(env_yaml, "control_mode")[1].strip()
    if mode not in _CONTROL_MODES:
        raise SystemExit(f"{env_yaml}: control_mode {mode!r}; the only modes are {_CONTROL_MODES}")
    dec_key = "osc_decimation" if mode == "osc" else "decimation"
    dec = _int_value(env_yaml, dec_key, *_one_top_level(env_yaml, dec_key))
    if dec < 1:
        raise SystemExit(f"{env_yaml}: '{dec_key}:' must be >= 1, got {dec}")
    return dt * dec


_RL_GAMES_LOSSES_AT_STEP = ("losses/a_loss", "losses/c_loss", "losses/entropy", "losses/bounds_loss")
_RL_GAMES_AT_EPOCH_END = ("losses/cval_loss", "info/cval_lr")


def rl_games_x(tag: str, step: int, frames_per_epoch: int) -> int | None:
    """Total env steps behind one rl_games scalar, by its tag family (module
    docstring); None for a ``*/time`` tag. An unknown family is refused."""
    if tag.endswith("/time"):
        return None
    if tag.startswith("Episode/") or tag.endswith("/iter"):
        return step * frames_per_epoch
    if tag in _RL_GAMES_AT_EPOCH_END:
        return step
    if (tag.startswith(("info/", "performance/")) or tag in _RL_GAMES_LOSSES_AT_STEP
            or tag.endswith("/step")):
        return step + frames_per_epoch
    raise SystemExit(f"tag {tag!r}: no rl_games step-axis rule (run_axis.py docstring); add one, never guess")


def check_rl_games_epochs(pairs, frames_per_epoch: int) -> None:
    """``pairs`` = (step, ``info/epochs`` value) of a CSV. Every row must sit
    at (epoch - 1) * F; else the factor from the yaml is not the run's."""
    pairs = list(pairs)
    bad = [(s, e) for s, e in pairs if s != (int(e) - 1) * frames_per_epoch]
    if not pairs or bad:
        raise SystemExit(f"info/epochs rows disagree with F = {frames_per_epoch}: "
                         f"{'no rows' if not pairs else f'first {bad[:3]} (step, epoch)'}")


def _scene_num_envs(env_yaml) -> int:
    # hits: (line number, top-level block, directly under that block, value)
    hits = []
    block, child = None, None
    for no, line in enumerate(_lines(env_yaml), 1):
        m = _KEY.match(line)
        if m is None:
            continue
        indent = len(m.group("indent"))
        if indent == 0:
            block, child = m.group("key"), None
        elif child is None:
            child = indent
        if m.group("key") == "num_envs":
            hits.append((no, block, indent == child, m.group("rest")))
    under_scene = [h for h in hits if h[1] == "scene" and h[2]]
    if len(hits) != 1 or len(under_scene) != 1:
        where = ", ".join(f"line {h[0]} in block {h[1]!r}" for h in hits) or "none"
        raise SystemExit(f"{env_yaml}: expected exactly one 'num_envs:', directly under the top-level "
                         f"'scene:' block; found {len(hits)} ({where})")
    return _int_value(env_yaml, "num_envs", under_scene[0][0], under_scene[0][3])


def env_steps(it: int, num_steps_per_env: int, num_envs: int) -> int:
    """Total env steps (all envs summed) behind the scalars row ``step = it``."""
    return (it + 1) * num_steps_per_env * num_envs


def wall_time_line(steps: list[int], walls: list[float]) -> str:
    """The one line of text wall time gets (D-195): last minus first
    ``wall_time`` of the ITERATION rows the caller passes (never the
    ``*/time``-only rows, whose step is a clock, not an iteration)."""
    if not steps:
        return "wall time: no iteration rows"
    return (f"wall time {(walls[-1] - walls[0]) / 3600:.2f} h over {len(steps)} iterations "
            f"(steps {steps[0]}..{steps[-1]}, last minus first wall_time)")


# Self-test (D-080): checks on built data
CHECKS = (
    "real RT-235s1 files read (128, 256)",
    "a second num_envs under another block is refused",
    "a lone num_envs under another block is refused",
    "a num_envs nested deeper under scene is refused",
    "num_envs twice under scene is refused",
    "a missing num_envs is refused",
    "a missing num_steps_per_env is refused",
    "an indented num_steps_per_env is not the top-level key",
    "env_steps(0, 128, 256) == 32768",
    "env_steps(399, 128, 256) == 13107200",
    "wall_time_line: last minus first over the rows passed",
    "rl_games: horizon_length under params.config read (128, 256)",
    "rl_games: a horizon_length outside params.config is refused",
    "rl_games: the library is told from the agent.yaml (both ways)",
    "rl_games_x: epoch 1 lands at F in all three families",
    "rl_games_x: an unknown tag family is refused",
    "check_rl_games_epochs: rows off F are refused",
    "read_policy_dt: joint_pd -> sim.dt x decimation, a second dt refused",
    "real RT-240c scalars.csv: every tag has a rule, info/epochs fits F = 32768",
    "read_policy_dt: osc -> sim.dt x osc_decimation (dumped decimation 2 ignored); "
    "missing or unknown control_mode refused",
    "real RT-242s1_env.yaml (OSC): read_policy_dt == 1/15 s",
)
_REAL = (_REPO / "rt_logs" / "RT-235s1" / "RT-235s1_agent.yaml",
         _REPO / "rt_logs" / "RT-235s1" / "RT-235s1_env.yaml")
_REAL_OSC_ENV = _REPO / "rt_logs" / "RT-242s" / "RT-242s1_env.yaml"
_REAL_RLG_CSV = _REPO / "rt_logs" / "RT-240c" / "RT-240c_scalars.csv"


def _real_rl_games_csv(ns: dict) -> bool | None:
    """Every tag of the real RT-240c CSV gets an x (or None for */time), and
    its info/epochs rows fit F = 128 * 256. None when the file is absent."""
    import csv

    if not _REAL_RLG_CSV.is_file():
        return None
    with open(_REAL_RLG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ns["check_rl_games_epochs"](((int(float(r["step"])), float(r["info/epochs"])) for r in rows
                                 if r.get("info/epochs")), 32768)
    for r in rows:
        for tag, v in r.items():
            if tag not in ("step", "wall_time") and v:
                ns["rl_games_x"](tag, int(float(r["step"])), 32768)
    return True


def _run_checks(ns: dict, tmp: pathlib.Path) -> dict:
    """``{check: True | False | None}``; None = the git-ignored real files
    are not on this machine (reported as SKIPPED, never as PASS)."""
    rf, es = ns["read_factor"], ns["env_steps"]
    agent = tmp / "agent.yaml"
    agent.write_text("seed: 1\nnum_steps_per_env: 128\nmax_iterations: 400\n", encoding="utf-8")

    def env(name, body):
        p = tmp / f"{name}.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    def refused(a, e, key):
        try:
            rf(a, e)
            return False
        except SystemExit as exc:
            return key in str(exc) and pathlib.Path(e if key == "num_envs" else a).name in str(exc)
        except Exception:   # noqa: BLE001 - a crash is not a refusal
            return False

    def value(f):
        try:
            return f()
        except BaseException:   # noqa: BLE001 - NotImplementedError, SystemExit alike
            return False

    scene = "scene:\n  num_envs: 256\n  env_spacing: 3.0\n"
    res = {}
    res[CHECKS[0]] = (None if not all(p.is_file() for p in _REAL)
                      else value(lambda: rf(*_REAL) == (128, 256)))
    res[CHECKS[1]] = refused(agent, env("second", "viewer:\n  num_envs: 4\n" + scene), "num_envs")
    res[CHECKS[2]] = refused(agent, env("lone", "viewer:\n  num_envs: 4\nscene:\n  env_spacing: 3.0\n"), "num_envs")
    res[CHECKS[3]] = refused(agent, env("deep", "scene:\n  robot:\n    num_envs: 4\n  env_spacing: 3.0\n"),
                             "num_envs")
    res[CHECKS[4]] = refused(agent, env("twice", scene + "  num_envs: 512\n"), "num_envs")
    res[CHECKS[5]] = refused(agent, env("none", "scene:\n  env_spacing: 3.0\n"), "num_envs")
    ok_env = env("ok", "seed: 1\n" + scene + "events: null\n")
    no_n = tmp / "no_n.yaml"
    no_n.write_text("seed: 1\nmax_iterations: 400\n", encoding="utf-8")
    res[CHECKS[6]] = refused(no_n, ok_env, "num_steps_per_env")
    deep_n = tmp / "deep_n.yaml"
    deep_n.write_text("algorithm:\n  num_steps_per_env: 16\n", encoding="utf-8")
    res[CHECKS[7]] = refused(deep_n, ok_env, "num_steps_per_env")
    res[CHECKS[8]] = value(lambda: es(0, 128, 256) == 32768)
    res[CHECKS[9]] = value(lambda: es(399, 128, 256) == 13107200)
    res[CHECKS[10]] = value(lambda: ns["wall_time_line"]([0, 1, 2], [100.0, 1900.0, 7300.0])
                            == "wall time 2.00 h over 3 iterations (steps 0..2, last minus first wall_time)")
    rlg = tmp / "rlg_agent.yaml"
    rlg.write_text("params:\n  seed: 1\n  config:\n    name: x\n    horizon_length: 128\n"
                   "    central_value_config:\n      minibatch_size: 512\n", encoding="utf-8")
    res[CHECKS[11]] = value(lambda: ns["read_factor_rl_games"](rlg, ok_env) == (128, 256))
    wrong = tmp / "rlg_wrong.yaml"
    wrong.write_text("params:\n  config:\n    central_value_config:\n      horizon_length: 128\n", encoding="utf-8")
    try:
        ns["read_factor_rl_games"](wrong, ok_env)
        res[CHECKS[12]] = False
    except SystemExit as exc:
        res[CHECKS[12]] = "horizon_length" in str(exc)
    except Exception:   # noqa: BLE001
        res[CHECKS[12]] = False
    res[CHECKS[13]] = value(lambda: ns["read_run_factor"](rlg, ok_env) == ("rl_games", (128, 256))
                            and ns["read_run_factor"](agent, ok_env) == ("rsl_rl", (128, 256)))
    F = 128 * 256
    x = ns["rl_games_x"]
    res[CHECKS[14]] = value(lambda: x("Episode/success_rate", 1, F) == F and x("rewards/iter", 1, F) == F
                            and x("losses/a_loss", 0, F) == F and x("info/epochs", 0, F) == F
                            and x("rewards/step", 0, F) == F and x("losses/cval_loss", F, F) == F
                            and x("info/cval_lr", F, F) == F and x("rewards/time", 87, F) is None)
    try:
        x("scores/mean", 0, F)
        res[CHECKS[15]] = False
    except SystemExit as exc:
        res[CHECKS[15]] = "scores/mean" in str(exc)
    except Exception:   # noqa: BLE001
        res[CHECKS[15]] = False
    chk = ns["check_rl_games_epochs"]
    try:
        chk([(0, 1.0), (F, 2.0), (2 * F, 3.0)], F)
        good = True
    except BaseException:   # noqa: BLE001
        good = False
    try:
        chk([(0, 1.0), (F, 2.0)], 2 * F)
        res[CHECKS[16]] = False
    except SystemExit:
        res[CHECKS[16]] = good
    except Exception:   # noqa: BLE001
        res[CHECKS[16]] = False
    def dt_refused(p):
        try:
            ns["read_policy_dt"](p)
            return False
        except SystemExit:
            return True
        except Exception:   # noqa: BLE001 - a crash is not a refusal
            return False

    # shaped like the real dump (RT-242s1_env.yaml): decimation and
    # osc_decimation both top level, control_mode names the one that runs
    sim = "sim:\n  physics_prim_path: /x\n  dt: 0.008333333333333333\n  render_interval: 2\n"
    rest = "decimation: 2\nepisode_length_s: 4.266666666666667\n"
    tail = "osc_kp_pos: 100.0\nosc_decimation: 8\n" + scene
    dt_pd = env("dt_pd", sim + rest + "control_mode: joint_pd\n" + tail)
    dt_two = env("dt2", "sim:\n  dt: 0.01\ndecimation: 8\ncontrol_mode: joint_pd\nviewer:\n  dt: 0.02\n" + scene)
    res[CHECKS[17]] = value(lambda: abs(ns["read_policy_dt"](dt_pd) - 1 / 60) < 1e-12) and dt_refused(dt_two)
    res[CHECKS[18]] = value(lambda: _real_rl_games_csv(ns))
    dt_osc = env("dt_osc", sim + rest + "control_mode: osc\n" + tail)
    res[CHECKS[19]] = (value(lambda: abs(ns["read_policy_dt"](dt_osc) - 1 / 15) < 1e-12)
                       and dt_refused(env("dt_nomode", sim + rest + tail))
                       and dt_refused(env("dt_badmode", sim + rest + "control_mode: ik\n" + tail)))
    res[CHECKS[20]] = (None if not _REAL_OSC_ENV.is_file()
                       else value(lambda: abs(ns["read_policy_dt"](_REAL_OSC_ENV) - 1 / 15) < 1e-12))
    return res


MUTATIONS = (
    ("x-at-it-not-it-plus-1", "return (it + 1) * num_steps_per_env * num_envs",
     "return it * num_steps_per_env * num_envs", (CHECKS[8], CHECKS[9])),
    ("first-num_envs-anywhere", 'under_scene = [h for h in hits if h[1] == "scene" and h[2]]',
     "under_scene = hits[:1]", (CHECKS[2], CHECKS[3])),
    ("rl-games-losses-at-their-step", "return step + frames_per_epoch", "return step",
     (CHECKS[14],)),
    ("osc-reads-dumped-decimation", 'dec_key = "osc_decimation" if mode == "osc" else "decimation"',
     'dec_key = "decimation"', (CHECKS[19], CHECKS[20])),
)
_SELF_TEST_MARKER = "# Self-test (D-080): checks on built data"


def _self_test() -> int:
    import tempfile

    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    def _exec(text):
        ns = {"__name__": "run_axis_under_test", "__file__": __file__}
        exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
        return ns

    bad = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        base = _run_checks(_exec(src), tmp)
        for name in CHECKS:
            got = base[name]
            print(f"[run_axis]   {'SKIPPED (not on this machine)' if got is None else 'PASS' if got else 'FAIL'}"
                  f"  {name}")
            bad += got is False
        cut = src.index(_SELF_TEST_MARKER)
        head, tail = src[:cut], src[cut:]
        for mname, old, new, want in MUTATIONS:
            if head.count(old) != 1:
                print(f"[run_axis]   mutation {mname}: anchor matches {head.count(old)} times -- FAIL")
                bad += 1
                continue
            got = _run_checks(_exec(head.replace(old, new) + tail), tmp)
            flipped = {k for k, v in got.items() if v is False}
            ok = flipped == {w for w in want if base[w] is not None}   # a skipped real file cannot flip
            print(f"[run_axis]   mutation {mname}: {'PASS' if ok else 'FAIL'}"
                  + ("" if ok else f" (flipped {sorted(flipped)}, declared {sorted(want)})"))
            bad += not ok
    total = len(CHECKS) + len(MUTATIONS)
    skipped = sum(v is None for v in base.values())
    print(f"[run_axis] self-test: {total - bad - skipped}/{total}" + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    print(__doc__)

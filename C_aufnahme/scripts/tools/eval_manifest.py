# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The evaluation run's folder name and its ``eval_manifest.json``."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys

SCHEMA = "eval-manifest-v1"

MANIFEST_FIELDS: tuple[str, ...] = (
    "plan_commit",
    "code_commit",
    "code_dirty",
    "checkpoint_path",
    "checkpoint_sha256",
    "table_path",
    "table_sha256",
    "table_rows",
    "width",
    "tag",
    "task",
    "seed",
    "num_envs",
    "episode_cap_steps",
    "step_dt_s",
    "action_mode",
    "control_mode",
    "obs_wrench_mode",
    "noise",
    "versions",
    "normalizer_sha256_before",
    "normalizer_sha256_after",
    "policy_steps_run",
    "envs_finished_first_episode",
    "wall_time_s",
    "env_cfg",
)

MAY_BE_NONE = ("plan_commit", "normalizer_sha256_before", "normalizer_sha256_after")

WIDTHS = ("target", "nominal")
_TAG = re.compile(r"^[A-Za-z0-9-]+$")


class ManifestError(ValueError):
    """The manifest or the folder name cannot be built as asked."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_dirname(width: str, table_sha256: str, tag: str = "") -> str:
    """``eval_<width>_<sha8>[_<tag>]`` -- the run folder below the checkpoint's."""
    if width not in WIDTHS:
        raise ManifestError(f"width {width!r} is not one of {WIDTHS}")
    sha = str(table_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ManifestError(f"table sha256 is not 64 hex characters: {table_sha256!r}")
    name = f"eval_{width}_{sha[:8]}"
    if tag:
        if not _TAG.match(tag):
            raise ManifestError(f"tag {tag!r}: letters, digits and '-' only")
        name += f"_{tag}"
    return name


def count_table_rows(path) -> int:
    """Data rows of the table CSV (header and blank lines not counted) -- the same rule ``autodr.RowTable.from_csv`` reads with."""
    lines = pathlib.Path(path).read_bytes().decode("utf-8").splitlines()
    return sum(1 for line in lines[1:] if line.strip())


def git_commits(repo_dir, plan_file: str = "eval_plan.md") -> dict:
    """``{\"plan_commit\", \"code_commit\", \"code_dirty\"}`` of the clone at ``repo_dir``."""
    def _git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            raise ManifestError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout.strip()

    head = _git("rev-parse", "HEAD")
    plan = _git("log", "-1", "--format=%H", "--", plan_file)
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"plan_commit": plan or None, "code_commit": head, "code_dirty": bool(dirty)}


def build_manifest(**fields) -> dict:
    """The manifest dict, or ``ManifestError`` naming every wrong field."""
    got, want = set(fields), set(MANIFEST_FIELDS)
    if got != want:
        raise ManifestError(
            f"manifest fields differ: missing {sorted(want - got)}, "
            f"unknown {sorted(got - want)}"
        )
    unset = [k for k in MANIFEST_FIELDS if fields[k] is None and k not in MAY_BE_NONE]
    if unset:
        raise ManifestError(f"manifest fields without a value: {unset}")
    out = {"schema": SCHEMA}
    out.update({k: fields[k] for k in MANIFEST_FIELDS})
    return out


def write_manifest(run_dir, manifest: dict) -> pathlib.Path:
    path = pathlib.Path(run_dir) / "eval_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path


def _refuses(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ManifestError:
        return True
    return False


def _self_test() -> int:
    import tempfile

    full = {k: 0 for k in MANIFEST_FIELDS}
    sha = "ab" * 32
    checks: list[tuple[str, bool]] = []
    m = build_manifest(**full)
    checks.append(("a full field set builds, schema first",
                   list(m) == ["schema", *MANIFEST_FIELDS] and m["schema"] == SCHEMA))
    for k in MANIFEST_FIELDS:
        less = {a: b for a, b in full.items() if a != k}
        checks.append((f"dropping {k!r} is refused", _refuses(build_manifest, **less)))
    checks.append(("an unknown field is refused", _refuses(build_manifest, **full, extra=1)))
    for k in MANIFEST_FIELDS:
        holed = dict(full, **{k: None})
        checks.append((
            f"{k!r} = None is {'allowed' if k in MAY_BE_NONE else 'refused'}",
            _refuses(build_manifest, **holed) == (k not in MAY_BE_NONE),
        ))
    checks.append(("folder name", eval_dirname("target", sha) == "eval_target_abababab"))
    checks.append(("folder name with a tag",
                   eval_dirname("nominal", sha.upper(), "rep2") == "eval_nominal_abababab_rep2"))
    checks.append(("an unknown width is refused", _refuses(eval_dirname, "wide", sha)))
    checks.append(("a short hash is refused", _refuses(eval_dirname, "target", "abab")))
    checks.append(("a tag with a path separator is refused",
                   _refuses(eval_dirname, "target", sha, "a/b")))
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "t.csv"
        p.write_bytes(b"a,b\n1,2\n\n3,4\n")
        checks.append(("rows: header and blank lines are not counted", count_table_rows(p) == 2))
        checks.append(("sha256_file equals hashlib on the bytes",
                       sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()))
        out = write_manifest(tmp, dict(m, env_cfg={"s": slice(0, 3)}))
        back = json.loads(out.read_text(encoding="utf-8"))
        checks.append(("the manifest round-trips through JSON with every field",
                       list(back) == ["schema", *MANIFEST_FIELDS]))
    repo = pathlib.Path(__file__).resolve().parents[2]
    g = git_commits(repo)
    checks.append(("git_commits reads a 40-hex HEAD of this clone",
                   bool(re.fullmatch(r"[0-9a-f]{40}", g["code_commit"]))
                   and isinstance(g["code_dirty"], bool)))
    checks.append(("a folder that is no clone is refused",
                   _refuses(git_commits, pathlib.Path(tempfile.gettempdir()) / "no-such-clone")))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{'ALL ' + str(len(checks)) + ' CHECKS PASSED' if not failed else 'FAILED: ' + str(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(_self_test())
    raise SystemExit("usage: python scripts/tools/eval_manifest.py --self-test")

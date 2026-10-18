"""Copy the 15 evaluated D checkpoints into models, verifying archived SHA256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(d_root: Path) -> list[dict]:
    evaluation = d_root.parent / "auswertung/daten/endstudie_rl_games/evaluation"
    models = []
    for method, rt in (("autodr", "242"), ("fixeddr", "243"), ("nodr", "244")):
        block = "T-final-F" if method == "fixeddr" else "T-final"
        for seed in range(1, 6):
            policy = f"{method}_s{seed}"
            manifest = evaluation / block / policy / "eval_manifest.json"
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            original = PureWindowsPath(data["checkpoint_path"])
            expected = str(data["checkpoint_sha256"]).lower()
            run_id = f"RT-{rt}s{seed}"
            if (original.parent.name != "nn" or f"__{run_id}_" not in original.parent.parent.name
                    or not re.fullmatch(r"last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth", original.name)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError(f"Unexpected checkpoint identity in {manifest}")
            models.append({"policy": policy, "run_id": run_id, "source": Path(str(original)),
                           "sha256": expected,
                           "target": d_root / "models" / original.parent.parent.name / "nn" / original.name})
    return models


def import_models(d_root: Path, extra_roots: list[Path], check_only: bool = False) -> int:
    models = load_models(d_root)
    cache = {}

    def matches(path, expected):
        if not path.is_file():
            return False
        key = str(path.resolve())
        if key not in cache:
            cache[key] = sha256(path)
        return cache[key] == expected

    pending = []
    for model in models:
        target = model["target"]
        if target.exists():
            if not matches(target, model["sha256"]):
                raise ValueError(f"Existing target differs; not overwritten: {target}. For LFS pointers run git lfs pull.")
            model["found"] = target
        elif matches(model["source"], model["sha256"]):
            model["found"] = model["source"]
        else:
            pending.append(model)

    if pending:
        roots = extra_roots + [d_root / "logs/rl_games/ur5e_insertion",
                              d_root.parent.parent / "Sicherungen",
                              Path.home() / "Desktop/Alexander_Pett/view"]
        roots += list({model["source"].parents[2] for model in pending})
        candidates = set()
        explicit_files = set()
        for root in roots:
            if root.is_file():
                candidates.add(root); explicit_files.add(root)
            elif root.is_dir():
                candidates.update(root.rglob("*.pth"))
        for model in pending:
            marker = re.compile(re.escape(model["run_id"]) + r"(?!\d)", re.I)
            choices = [p for p in candidates if p in explicit_files or p.name == model["source"].name or marker.search(str(p))]
            choices.sort(key=lambda p: (p.name != model["source"].name, str(p)))
            for candidate in choices:
                if matches(candidate, model["sha256"]):
                    model["found"] = candidate
                    break

    missing = [m for m in models if "found" not in m]
    for model in models:
        print(f"{'OK' if 'found' in model else 'MISSING'} {model['policy']}: {model.get('found', model['source'])}")
    if missing:
        print(f"STOP: {len(missing)} checkpoint(s) missing or with different SHA256; nothing copied. "
              "Use --source-root <folder-or-file> for another location.")
        return 2
    total = sum(m["found"].stat().st_size for m in models)
    print(f"Verified 15/15 checkpoint hashes; {total / (1024 ** 3):.3f} GiB.")
    if check_only:
        return 0
    for model in models:
        target, source = model["target"], model["found"]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".partial", dir=target.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            shutil.copyfile(source, temporary)
            if sha256(temporary) != model["sha256"]:
                raise ValueError(f"Copy verification failed: {target}")
            if target.exists():
                raise ValueError(f"Target appeared during copying; not overwritten: {target}")
            temporary.rename(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    print("SUCCESS: 15/15 verified models in D_factory/models. Originals retained.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", default=[], help="Additional folder or checkpoint; repeatable.")
    parser.add_argument("--check-only", action="store_true", help="Verify availability and hashes without copying.")
    args = parser.parse_args()
    try:
        return import_models(Path(__file__).resolve().parents[1], args.source_root, args.check_only)
    except (OSError, ValueError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

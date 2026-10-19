"""Copy the selected original checkpoint and assets without changing their contents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_PROJECT = 'generalisation-angle-probe'
RUN = '2026-08-16_15-12-40_peg30mm_pocket32mm_offset20mm_tilt0-10deg_yaw+-45deg'
CHECKPOINT = 'model_7093.pt'
ROBOT = 'ur10e_peg_s30.usd'
TABLE = 'tisch_square_b32.usd'
ASSETS = Path("source/proxytask/proxytask/tasks/direct/proxytask/assets")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "Desktop/Alexander_Pett/temp/proxytask_gen" / SOURCE_PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="Check all files without copying.")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    training = Path("logs/rsl_rl/demo_insertion") / RUN
    model = Path("models") / RUN
    files = [(training / CHECKPOINT, model / CHECKPOINT),
             (ASSETS / "Robot" / ROBOT, ASSETS / "Robot" / ROBOT),
             (ASSETS / "Tisch" / TABLE, ASSETS / "Tisch" / TABLE)]
    for name in ("params/env.yaml", "params/agent.yaml", "demo_metrics.json"):
        if (source / training / name).is_file():
            files.append((training / name, model / name))
        else:
            print(f"[INFO] Original metadata not present: {training / name}")
    for folder, name in (("Robot", ROBOT), ("Tisch", TABLE)):
        for sidecar in (source / ASSETS / folder).glob(Path(name).stem + ".*.json"):
            relative = sidecar.relative_to(source)
            files.append((relative, relative))

    # Preflight all files before copying; never replace differing destination files.
    manifest = []
    for src, dst in files:
        original, target = source / src, destination / dst
        if not original.is_file():
            raise SystemExit(f"Missing original file: {original}")
        if original.stat().st_size == 0:
            raise SystemExit(f"Empty original file: {original}")
        with original.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit(f"Only a Git LFS pointer is present: {original}")
        digest = sha256(original)
        if target.exists() and sha256(target) != digest:
            raise SystemExit(f"Different file already present; not overwritten: {target}")
        manifest.append({"source": str(original), "file": dst.as_posix(),
                         "sha256": digest, "bytes": original.stat().st_size})
    if args.dry_run:
        print(f"[OK] {len(files)} files available; nothing copied.")
        return
    for entry in manifest:
        target = destination / entry["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(entry["source"], target)
        if sha256(target) != entry["sha256"]:
            raise SystemExit(f"Copy verification failed: {target}")
        print(f"[OK] {entry['file']}")
    provenance = {"run": RUN, "checkpoint": CHECKPOINT, "files": manifest}
    (destination / model / "HERKUNFT.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[OK] Original files copied and SHA-256 verified. GUI replay remains to be checked.")


if __name__ == "__main__":
    main()

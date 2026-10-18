"""Read-only C inventory; packages verified originals and evidence into a new ZIP.

No simulator, installation, conversion, source edit, or D write is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
import zipfile

EXPECTED = {'source_folder': 'Phase5_Robustheit_v1', 'models': [{'run_id': 'RT-206s1', 'folder': '09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1', 'checkpoint': 'model_1499.pt', 'sha256': 'b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394', 'method': 'autodr'}, {'run_id': 'RT-206s2', 'folder': '09-15_18-05-48__RT-206s2_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed2', 'checkpoint': 'model_1499.pt', 'sha256': '821794595d73469550d4cdc7ace841165a3aed16f662b69495b0576f4085c537', 'method': 'autodr'}, {'run_id': 'RT-206s3', 'folder': '09-16_02-12-05__RT-206s3_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed3', 'checkpoint': 'model_1499.pt', 'sha256': '8a3cf5f9540a0f0c71377e191d3f9b00a436322975b179391e798cb05d48bc3f', 'method': 'autodr'}, {'run_id': 'RT-206s4', 'folder': '09-16_15-10-35__RT-206s4_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed4', 'checkpoint': 'model_1499.pt', 'sha256': 'ea31716c1e3d55c9358c50828c394230fc00270d40842d3ae8c38cddba15aa0d', 'method': 'autodr'}, {'run_id': 'RT-206s5', 'folder': '09-17_00-05-01__RT-206s5_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed5', 'checkpoint': 'model_1499.pt', 'sha256': '92824ddd000b48c8e2e129e2577b333fbb773911ca93369ac6669539135dc403', 'method': 'autodr'}, {'run_id': 'RT-207s1', 'folder': '09-17_09-58-45__RT-207s1_offset5mm_currOFF_start+20mm_floor-30mm_seed1', 'checkpoint': 'model_1499.pt', 'sha256': 'a444c39ca54e498e6d930175961cb0278ecf098467ec20602b20fe44b895143d', 'method': 'nodr'}, {'run_id': 'RT-207s2', 'folder': '09-17_16-03-29__RT-207s2_offset5mm_currOFF_start+20mm_floor-30mm_seed2', 'checkpoint': 'model_1499.pt', 'sha256': '6a471014f70d96d16f7a6afc7db843eef262a11d87f510d3f1f6fed2b75c4c3b', 'method': 'nodr'}, {'run_id': 'RT-207s3', 'folder': '09-17_22-41-29__RT-207s3_offset5mm_currOFF_start+20mm_floor-30mm_seed3', 'checkpoint': 'model_1499.pt', 'sha256': 'f1e40a3dbeb6f528f1b1869b45ae71cc27d59c6b1fc92a3acbcb54f8cd08d610', 'method': 'nodr'}, {'run_id': 'RT-207s4', 'folder': '09-18_12-38-12__RT-207s4_offset5mm_currOFF_start+20mm_floor-30mm_seed4', 'checkpoint': 'model_1499.pt', 'sha256': '8e194f8ee0bf4efebd13c5c741b3d40c66488d96d0d4902279ad9e366458ad8a', 'method': 'nodr'}, {'run_id': 'RT-207s5', 'folder': '09-18_20-38-33__RT-207s5_offset5mm_currOFF_start+20mm_floor-30mm_seed5', 'checkpoint': 'model_1499.pt', 'sha256': '9019d28e1b164f068f0b57146b8c6818c977e0e4e8fd785854d1710eebf2174c', 'method': 'nodr'}, {'run_id': 'RT-231s1', 'folder': '09-24_02-23-58__RT-231s1_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed1', 'checkpoint': 'model_1499.pt', 'sha256': '51e9e74dc1e3aaf6302e6615b6c1da317a5f2a818bec8e443ea071ae359fe11d', 'method': 'fixeddr'}, {'run_id': 'RT-231s2', 'folder': '09-24_05-34-28__RT-231s2_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed2', 'checkpoint': 'model_1499.pt', 'sha256': 'd720cf0c17e41c8006ffe838eba0b8bd96e6cea5899624256694e31b2bb483b7', 'method': 'fixeddr'}, {'run_id': 'RT-231s3', 'folder': '09-24_08-07-07__RT-231s3_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed3', 'checkpoint': 'model_1499.pt', 'sha256': 'f3136fff1e458a824510c5440bc7af07eda8a909798f093bc80ffc3f2b695067', 'method': 'fixeddr'}, {'run_id': 'RT-231s4', 'folder': '09-24_11-18-12__RT-231s4_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed4', 'checkpoint': 'model_1499.pt', 'sha256': 'c860a0f895e8ef1ab033c84af4084c2d1cf20bca0dd47624b06baefedcffc7ed', 'method': 'fixeddr'}, {'run_id': 'RT-231s5', 'folder': '09-24_14-51-40__RT-231s5_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed5', 'checkpoint': 'model_1499.pt', 'sha256': '441b34d649f3dd33f9d890f4f5dca13f5d659b93076ade32736aad275fd00a50', 'method': 'fixeddr'}], 'assets': [{'path': 'Arbeitszelle/arbeitszelle_tische_v2.usd', 'sha256': '2df81f2489d0350bb24e3dce4c46a7fbeead2d8cb5f925c88a062a6f10a06ca1'}, {'path': 'Robot/fuegeteil.obj', 'sha256': 'cb6b6f860c945df63f0aa2f31c0f785e4f9200d5a7d236236f5b9c9a6963df1a'}, {'path': 'Robot/ur5e_tool.author.json', 'sha256': 'c15f50113c3f1d86d4d33f9fb8167bd2e13ae6c72ef18fdb4f34312bce0c4f78'}, {'path': 'Robot/ur5e_tool.usd', 'sha256': '94f7094daa539157d76a1834c346108b9d0f2005455eb3b04e7d3ab6b624936c'}, {'path': 'Werkzeug/aufnahme.obj', 'sha256': '9ad44691169b45cd2500f8ced84a634350711034862e58ff01631244db37e5f8'}, {'path': 'Werkzeug/Aufnahme_real_v1_mm.usd', 'sha256': '4ccff283dd3ee1bf7a00343d977cc2b9340ce3232509a25dec865fe1ecf4702a'}, {'path': 'Werkzeug/Aufnahme_real_v1_mm.verify.json', 'sha256': '7dcf0d9d3de3c6caa46e9d7c345368c8318b1c997b02a3f0aa6bfc9e1adbbd3b'}], 'training_fields': {'control_mode': 'osc', 'obs_wrench_mode': 'force', 'start_floor_m': -0.03, 'autodr_stall_buffers': 3, 'action_rate_scale': 0.0034, 'fixture_pos_noise_xy': 0.005, 'curriculum_enabled': False}}
ASSET_DIR = Path("source/insertion/insertion/tasks/direct/insertion/assets")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def collect(source, output_dir):
    source = source.resolve()
    run_root = source / "logs/rsl_rl/ur5e_insertion"
    if not run_root.is_dir():
        raise FileNotFoundError(f"C training directory missing: {run_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("C_Bestand_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".zip")
    report = {"source": str(source), "python": sys.executable, "versions": {},
              "models": [], "assets": [], "files": [], "problems": []}
    for package in ("numpy", "torch", "rsl-rl-lib", "isaaclab", "isaacsim", "insertion"):
        try:
            report["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["versions"][package] = None
    logs = sorted((source / "rt_logs").rglob("*.txt")) if (source / "rt_logs").is_dir() else []
    names = set()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3,
                         allowZip64=True) as archive:
        def add(path, name):
            path.resolve().relative_to(source)  # Do not follow links outside the original C folder.
            if name in names:
                return next(item for item in report["files"] if item["archive"] == name)
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            item = {"source": str(path), "archive": name, "bytes": size, "sha256": digest.hexdigest()}
            report["files"].append(item)
            names.add(name)
            return item

        for expected in EXPECTED["models"]:
            run_id = expected["run_id"]
            folder = run_root / expected["folder"]
            model = folder / expected["checkpoint"]
            entry = {**expected, "path": str(model), "status": "MISSING", "full_log_candidates": [],
                     "short_logs": [], "sidecar_present": (folder / "autodr_1499.json").is_file()}
            if model.is_file():
                actual = sha256(model)
                entry["actual_sha256"] = actual
                entry["status"] = "OK" if actual == expected["sha256"] else "HASH_MISMATCH"
                if entry["status"] == "OK":
                    info = add(model, f"models/{expected['folder']}/{expected['checkpoint']}")
                    if info["sha256"] != actual:
                        entry["status"] = "CHANGED_DURING_READ"
            if entry["status"] != "OK":
                report["problems"].append(f"{run_id}: model {entry['status']}")
            if not entry["sidecar_present"]:
                report["problems"].append(f"{run_id}: autodr_1499.json missing")
            if folder.is_dir():
                selected = [p for p in folder.iterdir() if p.is_file() and
                            (p.name == "autodr_1499.json" or
                             (p.suffix.lower() in (".csv", ".txt", ".log", ".yaml", ".yml", ".json")
                              and not p.name.startswith("autodr_")))]
                if (folder / "params").is_dir():
                    selected += [p for p in (folder / "params").rglob("*") if p.is_file()]
                for path in sorted(set(selected)):
                    add(path, f"training/{run_id}/{path.relative_to(folder).as_posix()}")
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(run_id) + r"(?![A-Za-z0-9])")
            candidates = [p for p in logs if pattern.search(str(p.relative_to(source / "rt_logs")))]
            if folder.is_dir():
                candidates += [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".log")]
            for path in sorted(set(candidates)):
                relative = path.relative_to(source).as_posix()
                add(path, "logs/" + relative)
                key = "short_logs" if ".short." in path.name.lower() else "full_log_candidates"
                entry[key].append(relative)
            if not entry["full_log_candidates"]:
                report["problems"].append(f"{run_id}: no full console-log candidate found")
            report["models"].append(entry)
            print(f"{run_id}: {entry['status']}; full-log candidates: {len(entry['full_log_candidates'])}; sidecar: {entry['sidecar_present']}", flush=True)

        for expected in EXPECTED["assets"]:
            path = source / ASSET_DIR / expected["path"]
            entry = {**expected, "status": "MISSING"}
            if path.is_file():
                info = add(path, "assets/" + expected["path"])
                entry["actual_sha256"] = info["sha256"]
                entry["status"] = "IDENTICAL_TO_D" if info["sha256"] == expected["sha256"] else "DIFFERS_FROM_D"
            report["assets"].append(entry)
            if entry["status"] != "IDENTICAL_TO_D":
                report["problems"].append(f"Asset {expected['path']}: {entry['status']}")

        # Include the original C source needed to resolve any end-state differences.
        for relative in ("source/insertion", "scripts/rsl_rl", "scripts/tools"):
            folder = source / relative
            if folder.is_dir():
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".toml", ".yaml") and "__pycache__" not in path.parts:
                        add(path, "code/" + path.relative_to(source).as_posix())
        for relative in ("scripts/run_eval.ps1", "scripts/eval_collect.py", "scripts/eval_stats.py",
                         "scripts/check_eval_conditions.py", "eval_plan.md"):
            path = source / relative
            if path.is_file():
                add(path, "code/" + relative)
        report["verified_models"] = sum(item["status"] == "OK" for item in report["models"])
        report["runs_with_full_log_candidates"] = sum(bool(item["full_log_candidates"]) for item in report["models"])
        archive.writestr("bericht.json", json.dumps(report, ensure_ascii=False, indent=2))
        archive.writestr("erwartete_modelle.json", json.dumps(EXPECTED, ensure_ascii=False, indent=2))

    # Verify the transfer archive, not just the source paths.
    with zipfile.ZipFile(output) as archive:
        for item in report["files"]:
            digest = hashlib.sha256()
            with archive.open(item["archive"]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"ZIP verification failed: {item['archive']}")
    print(f"\nVerified models: {report['verified_models']}/15")
    print(f"Runs with full-log candidates: {report['runs_with_full_log_candidates']}/15")
    for problem in report["problems"]:
        print("CHECK:", problem)
    print("ZIP verified:", output)
    return 0 if not report["problems"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "Desktop/Alexander_Pett/Zips")
    args = parser.parse_args()
    try:
        return collect(args.source, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

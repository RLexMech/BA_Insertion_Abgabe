"""Prepare separate GUI checkpoints and small excerpts of existing test tables.

Run in the existing training environment, before starting Isaac Sim.
Original files and installed packages are never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import math
from pathlib import Path
import pickletools
import shutil
import tempfile
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_new(source, target):
    """Copy a validated temporary file without replacing an existing file."""
    with target.open("xb") as output:
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
        except BaseException:
            output.close()
            target.unlink()
            raise


def checkpoint(source, target):
    import numpy as np
    import torch

    def convert(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, dict):
            result = copy.copy(value)
            result.clear()
            result.update((key, convert(item)) for key, item in value.items())
            return result
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if value is None or type(value) in (bool, int, float, str, bytes):
            return value
        raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

    def verify(expected, actual, path="checkpoint"):
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or expected.dtype != actual.dtype or expected.shape != actual.shape:
                raise ValueError(f"Tensor type or shape differs: {path}")
            left = expected.detach().contiguous().reshape(-1).view(torch.uint8)
            right = actual.detach().contiguous().reshape(-1).view(torch.uint8)
            if not torch.equal(left, right):
                raise ValueError(f"Tensor bytes differ: {path}")
            return 1
        if type(expected) is not type(actual):
            raise ValueError(f"Value type differs: {path}")
        if isinstance(expected, dict):
            if expected.keys() != actual.keys():
                raise ValueError(f"Keys differ: {path}")
            return sum(verify(item, actual[key], f"{path}.{key}") for key, item in expected.items())
        if isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                raise ValueError(f"Length differs: {path}")
            return sum(verify(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(expected, actual)))
        if isinstance(expected, float) and math.isnan(expected) and math.isnan(actual):
            return 0
        if expected != actual:
            raise ValueError(f"Value differs: {path}")
        return 0

    def verify_file(path, expected):
        # torch.save uses protocol 2 here: NumPy dependencies appear as GLOBAL opcodes.
        with zipfile.ZipFile(path) as archive:
            data = archive.read(next(name for name in archive.namelist() if name.endswith("/data.pkl")))
        if any(op.name == "GLOBAL" and arg.split(" ")[0].startswith("numpy")
               for op, arg, _ in pickletools.genops(data)):
            raise ValueError("View file still contains NumPy pickle dependencies")
        actual = torch.load(path, map_location="cpu", weights_only=False)
        tensors = verify(expected, actual)
        weights = verify(expected["model"], actual["model"], "model")
        if not weights:
            raise ValueError("No model tensors found")
        print(f"PASS: {weights} model tensors identical, including stored normalizers; {tensors} tensors checked in total.")

    before = sha256(source)
    original = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(original, dict) or not isinstance(original.get("model"), dict) or not original["model"]:
        raise ValueError("Expected an RL-Games checkpoint with a nonempty 'model' state dictionary")
    converted = convert(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_file(target, converted)
        action = "Reused verified view"
    else:
        with tempfile.TemporaryDirectory(prefix=".prepare-view-", dir=target.parent) as directory:
            temporary = Path(directory) / "view.pth"
            torch.save(converted, temporary, pickle_protocol=2)
            verify_file(temporary, converted)
            if sha256(source) != before:
                raise ValueError("Source changed during preparation")
            publish_new(temporary, target)
        action = "Created view"
    if sha256(source) != before:
        raise ValueError("Source changed during preparation")
    print(f"PASS: original SHA256 unchanged: {before}")
    print(f"{action}: {target}")


def table(source, target, first_row, count):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or len(rows[0]) != 18:
        raise ValueError("Expected an 18-column D evaluation table")
    data = rows[1:]
    if first_row < 0 or count < 1 or first_row + count > len(data):
        raise ValueError(f"Invalid selection: start={first_row}, count={count}, available={len(data)}")
    selected = data[first_row:first_row + count]
    if any(len(row) != 18 or not all(math.isfinite(float(value)) for value in row) for row in selected):
        raise ValueError("Invalid numeric table row")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows([rows[0], *selected])
    content = buffer.getvalue().encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f"Different file exists; choose another output path: {target}")
    else:
        with target.open("xb") as stream:
            stream.write(content)
    print(f"View table: {target}; original row IDs {first_row}..{first_row + count - 1}; use --num_envs {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("checkpoint", "table"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("target", type=Path)
        if name == "table":
            command.add_argument("--first-row", type=int, default=0, help="Zero-based row ID; header excluded")
            command.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.source.resolve() == args.target.resolve():
            raise ValueError("Source and target must be different files")
        if args.command == "checkpoint":
            checkpoint(args.source, args.target)
        else:
            table(args.source, args.target, args.first_row, args.count)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        parser.exit(1, f"STOP: {exc}\n")


if __name__ == "__main__":
    main()

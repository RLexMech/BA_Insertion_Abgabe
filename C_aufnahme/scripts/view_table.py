"""Select unchanged C evaluation rows for a small GUI preview."""
import argparse
import csv
import io
import math
from pathlib import Path

def table(source, target, first_row, count):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or len(rows[0]) != 16:
        raise ValueError("Expected an 16-column C evaluation table")
    data = rows[1:]
    if first_row < 0 or count < 1 or first_row + count > len(data):
        raise ValueError(f"Invalid selection: start={first_row}, count={count}, available={len(data)}")
    selected = data[first_row:first_row + count]
    if any(len(row) != 16 or not all(math.isfinite(float(value)) for value in row) for row in selected):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--first-row", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.source.resolve() == args.target.resolve():
            raise ValueError("Source and target must be different files")
        table(args.source, args.target, args.first_row, args.count)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"STOP: {exc}\n")

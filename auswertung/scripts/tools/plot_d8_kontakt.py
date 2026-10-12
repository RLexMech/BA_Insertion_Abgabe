"""Frames 4-8 of the D8 still sequence plus one between the first two, enlarged, labelled (a)-(f) only.

Reads the frames straight from rt_logs/D8/D8_RT-243s4_fuegen.mkv (ffmpeg on PATH).
Times were found by matching the tiles of docs/figures/standbilder/D8_RT-243s4_bildfolge.png
against the video (tile crop = 440x400 at (310, 170), mean squared error 0.0 per tile).
Contact marks and descriptions are added by hand in the thesis (user 2026-10-08).

    python scripts/tools/plot_d8_kontakt.py
"""
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "rt_logs/D8/D8_RT-243s4_fuegen.mkv"
OUT = REPO / "docs/figures/standbilder/D8_RT-243s4_kontakt"

# crop in full-frame pixels: the tile crop (310, 170) narrowed to part and fixture
CROP = (310 + 40, 170 + 90, 310 + 300, 170 + 330)

# video time [s] of tiles 4-8 of the 8-frame sequence, plus 198.240 s between the first two:
# the part on the rim with its orientation changed against (a), chosen by eye from 12 frames
# between 193.833 and 210.000 s (user 2026-10-12, six panels like the C sequence)
TIMES = [193.833, 198.240, 210.000, 225.000, 237.733, 251.933]


def grab(t: float) -> Image.Image:
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(VIDEO), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True).stdout
    return Image.open(BytesIO(png)).convert("RGB").crop(CROP)


def main() -> int:
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4))
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, t) in enumerate(zip(axes.flat, TIMES)):
        ax.imshow(grab(t))
        ax.set_title(f"({chr(ord('a') + i)})", fontsize=10, pad=3)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT.with_suffix(f".{ext}"), **kw)
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Still sequence of the Experiment C final policy (RT-206s1) from the RT-272b recording, labelled (a)-(f) only.

Same form as `plot_d8_kontakt.py` (Abb. 5-4): crops of one episode, 2 x 3 grid, Arial 10.
Reads the frames straight from rt_logs/RT-272/RT-272b_C_ausrichten.mkv (ffmpeg on PATH).

Episode: the FIRST of the nine in the recording (39.3-45.6 s), which is trace episode 0
of rt_logs/RT-272/RT-272b_trace.json. The recording's nine episode starts were found
as frame-difference jumps of ~47 (39.3, 45.6, 52.1, 58.0, 64.3, 71.5, 77.3, 83.8, 90.6 s),
nine = the nine complete trace episodes. The times below were chosen by eye; a frame
is NOT tied to one trace row (the GUI step time is not constant).

    python scripts/tools/plot_c_ausrichten.py
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
VIDEO = REPO / "rt_logs/RT-272/RT-272b_C_ausrichten.mkv"
OUT = REPO / "docs/figures/standbilder/RT-272b_C_ausrichten"

# crop in full-frame pixels (1600 x 900 recording), part and fixture; D8 uses 260 x 240 too.
# Top at y 280: above it sits the corner of Isaac's Simulation Settings panel.
CROP = (330, 280, 330 + 260, 280 + 240)

# video time [s]: start, descent, just above the opening, at the opening, half way in
# (44.3 s, user 2026-10-12), seated
TIMES = [39.6, 41.2, 42.4, 42.8, 44.3, 45.3]


def grab(t: float) -> Image.Image:
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(VIDEO), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True).stdout
    return Image.open(BytesIO(png)).convert("RGB").crop(CROP)


def main() -> int:
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
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

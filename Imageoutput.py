import glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS – just tweak these
# ──────────────────────────────────────────────────────────────────────────────

PATTERN   = r"H:\algorithm\NSW1m\a4x4\npy\dem_aoi_*_slope.npy"
OUT_DIR   = "slope_warm"
os.makedirs(OUT_DIR, exist_ok=True)

# Display range in degrees
VMIN, VMAX = 0, 60

# Number of ticks on the legend
NUM_TICKS = 7

# Build a 3-color continuous ramp from gray → yellow → red
cmap = LinearSegmentedColormap.from_list(
    "gray_yellow_red",
    [
        (0.0, "#d3d3d3"),    # at the bottom (0%) → light gray
        (0.5, "#ffffb2"),    # halfway (50%) → yellow
        (1.0, "#e31a1c"),    # at the top (100%) → red
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# LOOP THROUGH YOUR .npy SLOPE ARRAYS
# ──────────────────────────────────────────────────────────────────────────────

for npy_fp in sorted(glob.glob(PATTERN)):
    name  = os.path.splitext(os.path.basename(npy_fp))[0]
    slope = np.load(npy_fp)

    # 1) Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    im = ax.imshow(slope, cmap=cmap, vmin=VMIN, vmax=VMAX)
    ax.set_axis_off()
    ax.set_title(f"{name} – Slope (°)", pad=12, fontsize=12)

    # 2) Colorbar with a handful of ticks
    ticks = np.linspace(VMIN, VMAX, NUM_TICKS)
    cbar  = fig.colorbar(im, ax=ax,
                         fraction=0.04, pad=0.02,
                         ticks=ticks)
    cbar.set_label("Slope (°)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # 3) Save and clean up
    out_png = os.path.join(OUT_DIR, f"{name}_warm.png")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print("Wrote:", out_png)


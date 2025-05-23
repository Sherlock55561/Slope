import os
import rasterio
import numpy as np
from scipy.ndimage import convolve, generic_filter
from algorithmethod import horn_slope, zeventho_slope

# ————————————————
# Helper to load a DEM + extract pixel size
# ————————————————
def load_dem(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        dx = src.transform.a
        dy = -src.transform.e
        nodata = src.nodata
    return arr, dx, dy, nodata

# ————————————————
# Filepaths (edit as needed)
# ————————————————
REF_DEM  = r"H:/algorithm/USGS_1M_17_x58y391_NC_Phase_4_CentralWestNC_GEIGER_A16.tif"
TEST_DEM = r"H:/algorithm/10_projected.tif"

# ————————————————
# Load both DEMs
# ————————————————
ref_dem, dx1, dy1, _     = load_dem(REF_DEM)
test_dem, dx2, dy2, nodata = load_dem(TEST_DEM)

print("REF dx,dy =", dx1, dy1)    # expect 1.0, 1.0
print("TEST dx,dy=", dx2, dy2)    # expect ~10.0, 10.0

# ————————————————
# Mask out NoData in test_dem
# ————————————————
test_dem = np.where(test_dem == nodata, np.nan, test_dem)

# ————————————————
# Auto-sample a 100×100 block within the valid footprint
# ————————————————
DO_SAMPLE = False
SAMPLE_SIZE = 100

if DO_SAMPLE:
    valid = ~np.isnan(test_dem)
    rows, cols = np.where(valid)
    if len(rows) < SAMPLE_SIZE*SAMPLE_SIZE:
        raise RuntimeError("Not enough valid pixels to sample!")
    # center your sample on the median valid pixel
    center_r = int(np.median(rows))
    center_c = int(np.median(cols))
    r0 = max(0, min(center_r - SAMPLE_SIZE//2, test_dem.shape[0]-SAMPLE_SIZE))
    c0 = max(0, min(center_c - SAMPLE_SIZE//2, test_dem.shape[1]-SAMPLE_SIZE))
    r1, c1 = r0 + SAMPLE_SIZE, c0 + SAMPLE_SIZE

    ref_dem  = ref_dem [r0:r1, c0:c1]
    test_dem = test_dem[r0:r1, c0:c1]
    print(f"Sampling window: rows {r0}:{r1}, cols {c0}:{c1}")
else:
    print("Running full DEM…")

# ————————————————
# 1) Horn on REF via convolution
# ————————————————
kx1 = np.array([[ 1, 2, 1],
                [ 0, 0, 0],
                [-1,-2,-1]]) / (8 * dx1)
ky1 = np.array([[ 1, 0,-1],
                [ 2, 0,-2],
                [ 1, 0,-1]]) / (8 * dy1)

dzdx_ref = convolve(ref_dem, kx1, mode='reflect')
dzdy_ref = convolve(ref_dem, ky1, mode='reflect')
slope_ref = np.degrees(np.arctan(np.hypot(dzdx_ref, dzdy_ref)))
print("REF slope min/max:", np.nanmin(slope_ref), np.nanmax(slope_ref))

# ————————————————
# 2) Prepare test_dem for convolution
# ————————————————
mean_val    = np.nanmean(test_dem)
test_filled = np.where(np.isnan(test_dem), mean_val, test_dem)

# Horn on test block
dzdx_test = convolve(test_filled, kx1, mode='reflect')
dzdy_test = convolve(test_filled, ky1, mode='reflect')
slope_horn = np.degrees(np.arctan(np.hypot(dzdx_test, dzdy_test)))

# ————————————————
# 3) Zevenbergen–Thorne on test block
# ————————————————
# new: run Zevenbergen on the nan-filled array
def zeven_filled_win(win):
    return zeventho_slope(win.reshape(3,3), dx2, dy2)

slope_zeven = generic_filter(
    test_filled,      # <–– the array where we replaced NaNs with mean_val
    zeven_filled_win,
    size=3,
    mode='reflect'
)


print("Horn test slope min/max:", np.nanmin(slope_horn), np.nanmax(slope_horn))
print("Zeven test slope min/max:", np.nanmin(slope_zeven), np.nanmax(slope_zeven))

# ————————————————
# 4) Save only the sample’s arrays
# ————————————————
BASE = os.path.dirname(__file__)
np.save(os.path.join(BASE, "slope_ref.npy"),   slope_ref)
np.save(os.path.join(BASE, "slope_horn.npy"),  slope_horn)
np.save(os.path.join(BASE, "slope_zeven.npy"), slope_zeven)
print("Slopes computed and saved to .npy files.")

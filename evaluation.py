import rasterio
import numpy as np
from scipy.ndimage import generic_filter
from algorithmethod import horn_slope, zeventho_slope
import matplotlib.pyplot as plt

# ————————————————
# Helper to load a DEM + extract pixel size
# ————————————————
def load_dem(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        dx = src.transform.a
        dy = -src.transform.e
    return arr, dx, dy

# ————————————————
# 1 m reference DEM (ground truth) and 10 m test DEM
# ————————————————
REF_DEM   = r"H:/algorithm/10.tif"
TEST_DEM  = r"H:/algorithm/USGS_1M_17_x58y391_NC_Phase_4_CentralWestNC_GEIGER_A16.tif"

# ————————————————
# Load both DEMs
# ————————————————
ref_dem,  dx1, dy1 = load_dem(REF_DEM)
test_dem, dx2, dy2 = load_dem(TEST_DEM)

DO_QUICK_TEST = True

if DO_QUICK_TEST:
    # either crop:
    ref_dem  = ref_dem[:5, :5]
    test_dem = test_dem[:5, :5]
    print("Running quick 5×5 crop test…")
else:
    print("Running full DEM…")
# ————————————————
# Compute reference slope (Horn’s method on 1 m DEM)
# ————————————————
def ref_horn_win(win):   return horn_slope(win, dx1, dy1)
slope_ref   = generic_filter(ref_dem,  ref_horn_win,  size=3)

# ————————————————
# Compute test slopes (on 10 m DEM)
# ————————————————
def test_horn_win(win):  return horn_slope(win, dx2, dy2)
def test_zeven_win(win): return zeventho_slope(win, dx2, dy2)

slope_horn  = generic_filter(test_dem, test_horn_win,  size=3)
slope_zeven = generic_filter(test_dem, test_zeven_win, size=3)

# ————————————————
# Save results for later evaluation
# ————————————————
np.save("slope_ref.npy",   slope_ref)
np.save("slope_horn.npy",  slope_horn)
np.save("slope_zeven.npy", slope_zeven)

print("Slopes computed and saved to .npy files.")

import rasterio
import numpy as np
from scipy.ndimage import generic_filter
from algorithmethod import horn_slope, zeventho_slope
import matplotlib.pyplot as plt

# Load DEM
dem_path = "G:/algorithm/NSW Government - Spatial Services/DEM/1 Metre/Katoomba201407-LID1-AHD_2566246_56_0002_0002_1m.tif"
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    dx = transform.a  # cell width
    dy = -transform.e 

def horn_window(window):
    return horn_slope(window, dx, dy)

def zeventho_window(window):
    return zeventho_slope(window, dx, dy)

slope_horn = generic_filter(dem, horn_window, size=3)
slope_zeven = generic_filter(dem, zeventho_window, size=3)

np.save("slope_horn.npy", slope_horn)
np.save("slope_zeven.npy", slope_zeven)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(slope_horn, cmap='terrain')
plt.title("Horn Slope (°)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(slope_zeven, cmap='terrain')
plt.title("Zevenbergen-Thorne Slope (°)")
plt.colorbar()

plt.tight_layout()
plt.show()
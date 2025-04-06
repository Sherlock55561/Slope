import numpy as np

def horn_slope(window, dx, dy):
    z = window.reshape((3, 3))
    dzdx = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) -
            (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * dx)
    dzdy = ((z[2, 2] + 2 * z[1, 2] + z[0, 2]) -
            (z[2, 0] + 2 * z[1, 0] + z[0, 0])) / (8 * dy)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)  # Return slope in degrees


def zeventho_slope(window, dx, dy):
    z = window.reshape((3, 3))
    p = (z[1, 2] - z[1, 0]) / (2 * dx)
    q = (z[0, 1] - z[2, 1]) / (2 * dy)
    slope_rad = np.arctan(np.sqrt(p**2 + q**2))
    return np.degrees(slope_rad)

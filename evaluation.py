import numpy as np
import pandas as pd

# Load saved arrays
horn = np.load("slope_horn.npy")
zeven = np.load("slope_zeven.npy")

# Clean and compare
horn_flat = horn.flatten()
zeven_flat = zeven.flatten()
matrix = np.vstack((horn_flat, zeven_flat)).T
matrix = matrix[~np.isnan(matrix).any(axis=1)]

df = pd.DataFrame(matrix, columns=["Horn", "Zevenbergen"])
df["Difference"] = np.abs(df["Horn"] - df["Zevenbergen"])

# Output
print(df.describe())
print(df[["Horn", "Zevenbergen"]].corr())
df.to_csv("slope_eval.csv", index=False)
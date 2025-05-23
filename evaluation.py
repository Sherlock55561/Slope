import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_flatten(base_dir):
    """
    Load the three slope arrays (already same-shape samples),
    mask out NaNs in the reference, and return flattened vectors.
    """
    ref   = np.load(os.path.join(base_dir, "slope_ref.npy"))
    horn  = np.load(os.path.join(base_dir, "slope_horn.npy"))
    zeven = np.load(os.path.join(base_dir, "slope_zeven.npy"))
  # if test slopes don’t match ref shape, resample them
    if horn.shape != ref.shape:
        from scipy.ndimage import zoom
        H, W = ref.shape
        horn = zoom(
            horn,
            (H/horn.shape[0], W/horn.shape[1]),
            order=1,
            mode='reflect'
        )
    if zeven.shape != ref.shape:
        from scipy.ndimage import zoom
        H, W = ref.shape
        zeven = zoom(
            zeven,
            (H/zeven.shape[0], W/zeven.shape[1]),
            order=1,
            mode='reflect'
        )

    mask = ~np.isnan(ref)
    return ref[mask].ravel(), horn[mask].ravel(), zeven[mask].ravel()

def compute_metrics(gt, est):
    return (
        np.sqrt(np.mean((gt - est)**2)),  # RMSE
        np.mean(np.abs(gt - est)),         # MAE
        np.corrcoef(gt, est)[0,1]          # Corr
    )

def save_overall_metrics(gt, horn, zeven, path="slope_metrics.csv"):
    rows = []
    for name, arr in [("Horn", horn), ("Zevenbergen", zeven)]:
        r, m, c = compute_metrics(gt, arr)
        rows.append((name, r, m, c))
    df = pd.DataFrame(rows, columns=["Method","RMSE","MAE","Corr"]).set_index("Method")
    df.to_csv(path)
    print("== Overall Metrics ==")
    print(df.to_string(float_format="%.3f"))
    return df

def sample_by_quantile(gt, horn, zeven, quantiles=(10,50,90)):
    """
    Select sample points at given GT percentiles, ensuring
    we always have at least three points.
    """
    valid = (gt > 0) & (~np.isnan(gt))
    gt_v   = gt[valid]
    horn_v = horn[valid]
    zeven_v= zeven[valid]

    # If too few points, fall back to min/median/max
    if gt_v.size < len(quantiles):
        qs = np.percentile(gt_v, [0,50,100])
        labels = ["min", "50th pct", "max"]
    else:
        qs = np.percentile(gt_v, quantiles)
        labels = [f"{p}th pct" for p in quantiles]

    idxs = [np.argmin(np.abs(gt_v - q)) for q in qs]
    df = pd.DataFrame({
        "GT":    gt_v[idxs],
        "Horn":  horn_v[idxs],
        "Zeven": zeven_v[idxs]
    }, index=labels)
    df["Err_Horn"]  = df["Horn"]  - df["GT"]
    df["Err_Zeven"] = df["Zeven"] - df["GT"]

    print("== Quantile Samples ==")
    print(df.to_string(float_format="%.3f"))
    return df

def plot_value_bar(samples, filename):
    fig, ax = plt.subplots(figsize=(6,4))
    samples[["GT","Horn","Zeven"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Slope (°)")
    ax.set_title("Slope Values by Sample")
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")

def plot_error_bar(samples, filename):
    err = samples[["Err_Horn","Err_Zeven"]].abs()
    fig, ax = plt.subplots(figsize=(6,4))
    err.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Absolute Error (°)")
    ax.set_title("Error by Sample")
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")

def plot_error_vs_gt(gt, est, method, filename):
    fig, ax = plt.subplots(figsize=(6,5))
    hb = ax.hexbin(gt, est-gt, gridsize=50, mincnt=1, cmap='Blues')
    fig.colorbar(hb, ax=ax, label="Count")
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel("GT Slope (°)")
    ax.set_ylabel(f"Error ({method}) (°)")
    ax.set_title(f"{method} Error vs GT")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")

def plot_bland_altman(horn, zeven, filename):
    avg  = 0.5*(horn + zeven)
    diff = horn - zeven
    md, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(avg, diff, s=2, alpha=0.5)
    ax.axhline(md, color='k')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_xlabel("Average Slope (°)")
    ax.set_ylabel("Horn − Zeven (°)")
    ax.set_title("Bland–Altman")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    gt, horn, zeven = load_and_flatten(BASE)

    # Overall metrics
    df_overall = save_overall_metrics(gt, horn, zeven, "slope_metrics.csv")

    # Quantile sampling
    samples_q = sample_by_quantile(gt, horn, zeven, quantiles=(25, 50, 75))
    plot_value_bar(samples_q,  "quantile_values.png")
    plot_error_bar(samples_q,  "quantile_errors.png")

    # Error distribution plots
    plot_error_vs_gt(gt, horn,    "Horn",       "horn_error_vs_gt.png")
    plot_error_vs_gt(gt, zeven,   "Zevenbergen","zeven_error_vs_gt.png")

    # Bland–Altman
    plot_bland_altman(horn, zeven, "bland_altman.png")

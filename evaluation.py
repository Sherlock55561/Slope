import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def load_and_resample(base_dir):
    ref = np.load(os.path.join(base_dir, "slope_ref.npy"))
    horn = np.load(os.path.join(base_dir, "slope_horn.npy"))
    zeven = np.load(os.path.join(base_dir, "slope_zeven.npy"))
    H, W = ref.shape
    horn_rs = zoom(horn,  (H/horn.shape[0],   W/horn.shape[1]), order=1, mode='reflect')
    zeven_rs = zoom(zeven,(H/zeven.shape[0], W/zeven.shape[1]),order=1, mode='reflect')
    mask = ~np.isnan(ref)
    return ref[mask].ravel(), horn_rs[mask].ravel(), zeven_rs[mask].ravel()


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
    print(df)
    return df


def sample_by_quantile(gt, horn, zeven, quantiles=(10,50,90), min_nonzero=0.1):
    mask = np.abs(gt) > min_nonzero
    gt_nz, hnz, znz = gt[mask], horn[mask], zeven[mask]
    qs = np.percentile(gt_nz, quantiles)
    idx = [np.argmin(np.abs(gt_nz - q)) for q in qs]
    labels = [f"{q}th pct" for q in quantiles]
    df = pd.DataFrame({"GT": gt_nz[idx], "Horn": hnz[idx], "Zeven": znz[idx]}, index=labels)
    df["Err_Horn"]  = df["Horn"]  - df["GT"]
    df["Err_Zeven"] = df["Zeven"] - df["GT"]
    print("== Quantile Samples ==")
    print(df)
    return df


def plot_value_bar(samples, filename):
    fig, ax = plt.subplots(figsize=(6,4))
    samples[["GT","Horn","Zeven"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Slope")
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
    ax.set_ylabel("|Error|")
    ax.set_title("Absolute Error by Sample")
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_error_vs_gt(gt, est, method, filename):
    fig, ax = plt.subplots(figsize=(6,5))
    hb = ax.hexbin(gt, est-gt, gridsize=200, mincnt=1, cmap='Blues')
    fig.colorbar(hb, ax=ax, label="Count")
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel("GT slope")
    ax.set_ylabel(f"Error ({method})")
    ax.set_title(f"{method} Error vs GT")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_bland_altman(horn, zeven, filename):
    avg = 0.5*(horn + zeven)
    diff = horn - zeven
    md = diff.mean(); sd = diff.std()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(avg, diff, s=1, alpha=0.3)
    ax.axhline(md, color='k')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_xlabel("Average slope")
    ax.set_ylabel("Horn - Zeven")
    ax.set_title("Bland-Altman")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    gt, horn, zeven = load_and_resample(BASE)

    # overall metrics
    save_overall_metrics(gt, horn, zeven, "slope_metrics.csv")

    # quantile sampling
    samples_q = sample_by_quantile(gt, horn, zeven)
    plot_value_bar(samples_q,  "quantile_values.png")
    plot_error_bar(samples_q,  "quantile_errors.png")

    # error distribution
    plot_error_vs_gt(gt, horn,    "Horn",       "horn_error_vs_gt.png")
    plot_error_vs_gt(gt, zeven,   "Zevenbergen","zeven_error_vs_gt.png")

    # method comparison
    plot_bland_altman(horn, zeven, "bland_altman.png")


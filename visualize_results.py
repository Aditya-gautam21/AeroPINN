"""
Standalone visualization script for AeroPINN.
Loads one simulation from the dataset, runs the trained model,
and produces ground truth / prediction / error comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves to files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.load_data import load_dataset
from src.model import PINN


# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/best.pt"
DATA_ROOT   = "data"
SPLIT       = "full_train"
N_SIMS      = 1          # visualise first simulation only
OUT_DIR     = "plots"
SUBSAMPLE   = 8000       # points to scatter (keeps plots fast & readable)
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PINN(fourier_embed_dim=64, fourier_scale=1.0, width=256, depth=6).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"]


def predict(model, X_np, stats, device):
    X_mean = torch.tensor(stats["X_mean"], dtype=torch.float32, device=device)
    X_std  = torch.tensor(stats["X_std"],  dtype=torch.float32, device=device)
    Y_mean = torch.tensor(stats["Y_mean"], dtype=torch.float32, device=device)
    Y_std  = torch.tensor(stats["Y_std"],  dtype=torch.float32, device=device)

    X = (torch.tensor(X_np, dtype=torch.float32, device=device) - X_mean) / X_std
    with torch.no_grad():
        pred_norm = model(X)
    return (pred_norm * Y_std + Y_mean).cpu().numpy()


def scatter(ax, x, y, c, title, cmap, vmin=None, vmax=None, label=""):
    sc = ax.scatter(x, y, c=c, s=2, cmap=cmap, vmin=vmin, vmax=vmax,
                    rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label=label)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_field_comparison(x, y, true, pred, field_idx, field_name, unit, out_path):
    """3-panel: ground truth | prediction | absolute error"""
    t = true[:, field_idx]
    p = pred[:, field_idx]
    err = np.abs(p - t)

    vmin, vmax = t.min(), t.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Field: {field_name}  |  RelErr = {err.mean()/(np.std(t)+1e-8):.4f}", fontsize=12)

    scatter(axes[0], x, y, t,   f"Ground Truth ({field_name})", "coolwarm", vmin, vmax, unit)
    scatter(axes[1], x, y, p,   f"Predicted ({field_name})",    "coolwarm", vmin, vmax, unit)
    scatter(axes[2], x, y, err, f"Absolute Error ({field_name})", "hot",    label=unit)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_velocity_magnitude(x, y, true, pred, out_path):
    """Speed magnitude comparison"""
    speed_true = np.sqrt(true[:, 0]**2 + true[:, 1]**2)
    speed_pred = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
    err = np.abs(speed_pred - speed_true)
    vmin, vmax = speed_true.min(), speed_true.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Velocity Magnitude |U|", fontsize=12)

    scatter(axes[0], x, y, speed_true, "Ground Truth |U|", "viridis", vmin, vmax, "m/s")
    scatter(axes[1], x, y, speed_pred, "Predicted |U|",    "viridis", vmin, vmax, "m/s")
    scatter(axes[2], x, y, err,        "Absolute Error",   "hot",     label="m/s")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_metrics_summary(true, pred, out_path):
    """Bar chart of MSE, MAE, RelErr per field"""
    fields = ["u (m/s)", "v (m/s)", "p (Pa)"]
    mse_vals, mae_vals, rel_vals = [], [], []

    for i in range(3):
        t, p = true[:, i], pred[:, i]
        mse = np.mean((p - t)**2)
        mae = np.mean(np.abs(p - t))
        rel = np.sqrt(mse) / (np.std(t) + 1e-8)
        mse_vals.append(mse)
        mae_vals.append(mae)
        rel_vals.append(rel)
        print(f"  {fields[i]:10s}  MSE={mse:.4e}  MAE={mae:.4e}  RelErr={rel:.4f}")

    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Error Metrics per Field", fontsize=12)

    for ax, vals, title, color in zip(
        axes,
        [mse_vals, mae_vals, rel_vals],
        ["MSE", "MAE", "Relative Error"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        bars = ax.bar(fields, vals, color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.3e}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_scatter_correlation(true, pred, out_path):
    """Predicted vs true scatter for each field"""
    fields = ["u (m/s)", "v (m/s)", "p (Pa)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Predicted vs Ground Truth", fontsize=12)

    for i, (ax, name) in enumerate(zip(axes, fields)):
        t, p = true[:, i], pred[:, i]
        ax.scatter(t, p, s=1, alpha=0.3, rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
        r2 = 1 - np.sum((p - t)**2) / (np.sum((t - t.mean())**2) + 1e-8)
        ax.set_title(f"{name}  R²={r2:.4f}")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model, stats = load_model(CHECKPOINT, device)
    print(f"  Checkpoint epoch: {torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['epoch']}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {N_SIMS} simulation(s) from '{SPLIT}'...")
    X_np, Y_np, _ = load_dataset(DATA_ROOT, split=SPLIT, max_simulations=N_SIMS)
    print(f"  Total points: {X_np.shape[0]:,}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\nRunning inference...")
    pred = predict(model, X_np, stats, device)

    # ── Subsample for scatter plots ───────────────────────────────────────────
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_np), size=min(SUBSAMPLE, len(X_np)), replace=False)
    x, y = X_np[idx, 0], X_np[idx, 1]
    true_s = Y_np[idx]
    pred_s = pred[idx]

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_field_comparison(x, y, true_s, pred_s, 0, "u", "m/s",
                          f"{OUT_DIR}/compare_u.png")
    plot_field_comparison(x, y, true_s, pred_s, 1, "v", "m/s",
                          f"{OUT_DIR}/compare_v.png")
    plot_field_comparison(x, y, true_s, pred_s, 2, "p", "Pa",
                          f"{OUT_DIR}/compare_p.png")
    plot_velocity_magnitude(x, y, true_s, pred_s,
                            f"{OUT_DIR}/compare_speed.png")

    print("\nMetrics (full dataset):")
    plot_metrics_summary(Y_np, pred, f"{OUT_DIR}/metrics_summary.png")

    plot_scatter_correlation(true_s, pred_s,
                             f"{OUT_DIR}/scatter_correlation.png")

    print(f"\nAll plots saved to '{OUT_DIR}/'")
    print("Files:", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()

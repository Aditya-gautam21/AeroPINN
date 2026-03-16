import matplotlib.pyplot as plt
import numpy as np


def plot_pressure(x, y, p, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    sc = ax.scatter(x, y, c=p, s=1, cmap="coolwarm")
    fig.colorbar(sc, ax=ax, label="Pressure (Pa)")
    ax.set_aspect("equal")
    ax.set_title("Pressure Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_velocity(x, y, u, v, save_path=None):
    speed = np.sqrt(u ** 2 + v ** 2)
    fig, ax = plt.subplots(figsize=(12, 4))
    sc = ax.scatter(x, y, c=speed, s=1, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Velocity magnitude (m/s)")
    ax.set_aspect("equal")
    ax.set_title("Velocity Magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_comparison(x, y, pred, true, field_idx=0, field_name="u", save_path=None):
    """Side-by-side comparison of predicted vs ground truth for one field."""
    vmin = min(pred[:, field_idx].min(), true[:, field_idx].min())
    vmax = max(pred[:, field_idx].max(), true[:, field_idx].max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, data, title in zip(
        axes,
        [true[:, field_idx], pred[:, field_idx], np.abs(pred[:, field_idx] - true[:, field_idx])],
        [f"Ground Truth ({field_name})", f"Predicted ({field_name})", f"Absolute Error ({field_name})"],
    ):
        sc = ax.scatter(x, y, c=data, s=1, cmap="coolwarm",
                        vmin=vmin if "Error" not in title else None,
                        vmax=vmax if "Error" not in title else None)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

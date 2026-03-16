import torch
import numpy as np

from .model import PINN
from .load_data import load_dataset
from .visualise import plot_comparison


def evaluate(
    checkpoint_path="checkpoints/best.pt",
    data_root="data",
    split="full_test",
    max_simulations=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    stats = ckpt["stats"]

    model = PINN(fourier_embed_dim=64, fourier_scale=1.0, width=256, depth=6).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    X_np, Y_np, _ = load_dataset(data_root, split=split, max_simulations=max_simulations)

    X_mean = torch.tensor(stats["X_mean"], dtype=torch.float32, device=device)
    X_std  = torch.tensor(stats["X_std"],  dtype=torch.float32, device=device)
    Y_mean = torch.tensor(stats["Y_mean"], dtype=torch.float32, device=device)
    Y_std  = torch.tensor(stats["Y_std"],  dtype=torch.float32, device=device)

    X = (torch.tensor(X_np, dtype=torch.float32, device=device) - X_mean) / X_std

    with torch.no_grad():
        pred_norm = model(X)

    # Denormalise
    pred = (pred_norm * Y_std + Y_mean).cpu().numpy()
    true = Y_np

    for i, name in enumerate(["u", "v", "p"]):
        mse = np.mean((pred[:, i] - true[:, i]) ** 2)
        mae = np.mean(np.abs(pred[:, i] - true[:, i]))
        rel = np.sqrt(mse) / (np.std(true[:, i]) + 1e-8)
        print(f"{name}: MSE={mse:.4e}  MAE={mae:.4e}  RelErr={rel:.4f}")

    x, y = X_np[:, 0], X_np[:, 1]
    for i, name in enumerate(["u", "v", "p"]):
        plot_comparison(x, y, pred, true, field_idx=i, field_name=name,
                        save_path=f"eval_{name}.png")
        print(f"Saved eval_{name}.png")


if __name__ == "__main__":
    evaluate()

import torch
import numpy as np
from pathlib import Path

from .model import PINN
from .physics import navier_stokes_loss
from .load_data import load_dataset


def train(
    data_root="data",
    split="full_train",
    max_simulations=None,
    epochs=10000,
    batch_size=4096,
    lr=1e-3,
    lambda_physics=0.1,
    checkpoint_dir="checkpoints",
    log_every=200,
    nu=1e-5,
):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load & normalise ──
    X_np, Y_np, stats = load_dataset(data_root, split=split, max_simulations=max_simulations)

    X_mean = torch.tensor(stats["X_mean"], dtype=torch.float32, device=device)
    X_std  = torch.tensor(stats["X_std"],  dtype=torch.float32, device=device)
    Y_mean = torch.tensor(stats["Y_mean"], dtype=torch.float32, device=device)
    Y_std  = torch.tensor(stats["Y_std"],  dtype=torch.float32, device=device)

    X = (torch.tensor(X_np, dtype=torch.float32, device=device) - X_mean) / X_std
    Y = (torch.tensor(Y_np, dtype=torch.float32, device=device) - Y_mean) / Y_std

    N = X.shape[0]

    # ── Model ──
    model = PINN(fourier_embed_dim=64, fourier_scale=1.0, width=256, depth=6).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_loss = float("inf")

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        X_batch = X[idx]
        Y_batch = Y[idx]

        pred = model(X_batch)
        data_loss = ((pred - Y_batch) ** 2).mean()

        physics_loss = navier_stokes_loss(model, X_batch, nu=nu)

        loss = data_loss + lambda_physics * physics_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>6} | loss={loss.item():.4e} "
                f"data={data_loss.item():.4e} "
                f"physics={physics_loss.item():.4e} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "stats": stats},
                f"{checkpoint_dir}/best.pt"
            )

    print(f"\nTraining complete. Best loss: {best_loss:.4e}")
    return model, stats


if __name__ == "__main__":
    train(
        data_root="data",
        split="full_train",
        max_simulations=5,
        epochs=10000,
        batch_size=4096,
        lr=1e-3,
        lambda_physics=0.1,
        nu=1e-5,
    )

import json
import pyvista as pv
import numpy as np
from pathlib import Path


def load_dataset(data_root, split="full_train", max_simulations=None):
    """
    Load AirFRANS dataset using the official manifest splits.

    Args:
        data_root: path to the data folder containing simulation dirs + manifest.json
        split: one of full_train, full_test, scarce_train, reynolds_train,
               reynolds_test, aoa_train, aoa_test
        max_simulations: limit number of simulations loaded (None = all)

    Returns:
        X: (N, 2) array of [x, y] coordinates
        Y: (N, 3) array of [u, v, p]
        stats: dict with mean/std for normalization
    """
    data_root = Path(data_root)

    with open(data_root / "manifest.json") as f:
        manifest = json.load(f)

    sim_names = manifest[split]
    if max_simulations is not None:
        sim_names = sim_names[:max_simulations]

    print(f"Loading {len(sim_names)} simulations from split '{split}'...")

    X_all, Y_all = [], []

    for name in sim_names:
        sim_dir = data_root / name
        vtu_files = list(sim_dir.glob("*internal.vtu"))
        if not vtu_files:
            print(f"  Warning: no internal.vtu found in {sim_dir}, skipping")
            continue

        mesh = pv.read(vtu_files[0])

        # Fields are already in point_data for AirFRANS .vtu files
        if "U" not in mesh.point_data or "p" not in mesh.point_data:
            mesh = mesh.cell_data_to_point_data()

        x = mesh.points[:, 0].astype(np.float32)
        y = mesh.points[:, 1].astype(np.float32)
        u = mesh.point_data["U"][:, 0].astype(np.float32)
        v = mesh.point_data["U"][:, 1].astype(np.float32)
        p = mesh.point_data["p"].astype(np.float32)

        X_all.append(np.stack([x, y], axis=1))
        Y_all.append(np.stack([u, v, p], axis=1))

    X = np.vstack(X_all)
    Y = np.vstack(Y_all)

    stats = {
        "X_mean": X.mean(axis=0),
        "X_std":  X.std(axis=0) + 1e-8,
        "Y_mean": Y.mean(axis=0),
        "Y_std":  Y.std(axis=0) + 1e-8,
    }

    print(f"Loaded {X.shape[0]:,} points total")
    return X, Y, stats

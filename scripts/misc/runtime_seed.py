from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None


def seed_torch_runtime(seed: int) -> None:
    """Seed Python, NumPy, and Torch so Torch-side training runs are repeatable."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

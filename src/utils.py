from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch as consistently as practical."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be considerably slower and some CUDA kernels
    # do not provide deterministic implementations. We therefore use the same
    # reproducibility policy as the paper experiments: deterministic CuDNN where
    # available, while allowing operations that have no deterministic variant.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device(spec: str = "auto") -> torch.device:
    spec = str(spec).strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Best-effort reproducibility across Python, NumPy, and PyTorch.

    Note: Some third-party libraries (e.g. Stim samplers) may not expose seed
    control; this function does not guarantee bit-for-bit identical data
    sampling in those cases.
    """
    if seed is None:
        raise ValueError("seed must be an int (not None)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


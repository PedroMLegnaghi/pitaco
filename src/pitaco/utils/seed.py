"""Seed every random number generator the project touches.

Fixed seeds make sampling, splits and training initialization repeatable. GPU kernels can still
be non-deterministic, so training runs may differ in the last decimals; splits and samples won't.
"""

import os
import random


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses (e.g. dataloader workers)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

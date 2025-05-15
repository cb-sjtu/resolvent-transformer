#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
from torch.utils.data import get_worker_info


def get_random_state_description(idx: int) -> str:
    """
    Get the random state description of the current sample.
    idx: the index of the current sample, argument of dataset.__getitem__
    """
    worker_id = get_worker_info().id
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    description = f"r/w: {rank}/{worker_id}, idx: {idx}, random state: {torch.randn(1).item()}"
    return description

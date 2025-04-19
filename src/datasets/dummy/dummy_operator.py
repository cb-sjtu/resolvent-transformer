import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets import dataset_utils as dsu


class DummyOperatorDataset(Dataset):
    def __init__(self, f_seq_len: int, g_seq_len: int, f_inout_dim: int, g_in_dim: int, g_out_dim: int):
        self.f_seq_len = f_seq_len
        self.g_seq_len = g_seq_len
        self.f_inout_dim = f_inout_dim
        self.g_in_dim = g_in_dim
        self.g_out_dim = g_out_dim

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        # get random state description in the beginning of __getitem__, to monitor the random state of each sample
        description = ""
        description += f"dataset: {self.__class__.__name__}, "
        description += dsu.get_random_state_description(idx)

        f_samples = torch.randn(1, self.f_seq_len, self.f_inout_dim)
        g_inputs = torch.randn(1, self.g_seq_len, self.g_in_dim)
        g_targets = torch.randn(1, self.g_seq_len, self.g_out_dim)

        data = {
            "f_samples": f_samples,
            "g_inputs": g_inputs,
        }

        label = g_targets

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label,
        }

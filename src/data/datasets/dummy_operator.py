from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import src.data.data_utils as du


@dataclass
class OperatorData(du.DataBase):
    f_samples: torch.Tensor = None
    g_inputs: torch.Tensor = None
    g_targets: torch.Tensor = None


class DummyOperatorDataset(Dataset):
    def __init__(self, f_seq_len: int, g_seq_len: int, f_inout_dim: int, g_in_dim: int, g_out_dim: int):
        self.f_seq_len = f_seq_len
        self.g_seq_len = g_seq_len
        self.f_inout_dim = f_inout_dim
        self.g_in_dim = g_in_dim
        self.g_out_dim = g_out_dim

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        f_samples = torch.randn(1, self.f_seq_len, self.f_inout_dim)
        g_inputs = torch.randn(1, self.g_seq_len, self.g_in_dim)
        g_targets = torch.randn(1, self.g_seq_len, self.g_out_dim)

        return OperatorData(f_samples=f_samples, g_inputs=g_inputs, g_targets=g_targets)

from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

import src.data.data_utils as du


@dataclass
class OperatorData(du.DataBase):
    f_samples: torch.Tensor = None
    g_inputs: torch.Tensor = None
    g_targets: torch.Tensor = None


class OperatorDataset(Dataset):
    def __init__(self, cfg: DictConfig = None):
        self.cfg = cfg

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        f_seq_len = self.cfg.get("f_seq_len")
        g_seq_len = self.cfg.get("g_seq_len")
        f_inout_dim = self.cfg.get("f_inout_dim")
        g_in_dim = self.cfg.get("g_in_dim")
        g_out_dim = self.cfg.get("g_out_dim")

        f_samples = torch.randn(1, f_seq_len, f_inout_dim)
        g_inputs = torch.randn(1, g_seq_len, g_in_dim)
        g_targets = torch.randn(1, g_seq_len, g_out_dim)

        return OperatorData(f_samples=f_samples, g_inputs=g_inputs, g_targets=g_targets)

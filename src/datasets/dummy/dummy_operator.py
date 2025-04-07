import torch
from torch.utils.data import Dataset

from src.datasets.data_utils import BaseLabelData, OperatorData


class DummyOperatorDataset(Dataset):
    # name is used for logging
    name: str = "dummy_operator_dataset"

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

        data = OperatorData(
            description=[f"data from {self.name}, idx: {idx}, random state: {torch.randn(1).item()}"],
            f_samples=f_samples,
            g_inputs=g_inputs,
        )
        label = BaseLabelData(description=[f"label from {self.name}"], label=g_targets)

        return {"data": data, "label": label}

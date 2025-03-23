from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import src.data.data_utils as du


@dataclass
class IconData(du.BaseData):
    cond_features: torch.Tensor = None
    qoi_features: torch.Tensor = None


class IconDataset(Dataset):
    # name is used for logging
    name: str = "dummy_icon_dataset"

    def __init__(self, demo_num: int, cond_shape: tuple[int, int, int], qoi_shape: tuple[int, int, int]):
        self.demo_num = demo_num
        self.cond_shape = cond_shape  # (cond_dim, cond_h, cond_w)
        self.qoi_shape = qoi_shape  # (qoi_dim, qoi_h, qoi_w)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        demo_cond_features = torch.randn(1, self.demo_num, self.cond_shape[0], self.cond_shape[1], self.cond_shape[2])
        demo_qoi_features = torch.randn(1, self.demo_num, self.qoi_shape[0], self.qoi_shape[1], self.qoi_shape[2])
        quest_cond_features = torch.randn(1, 1, self.cond_shape[0], self.cond_shape[1], self.cond_shape[2])
        quest_qoi_features = torch.randn(1, 1, self.qoi_shape[0], self.qoi_shape[1], self.qoi_shape[2])

        cond_features = torch.cat(
            (demo_cond_features, quest_cond_features), dim=1
        )  # (1, demo_num + 1, cond_dim, cond_h, cond_w)
        qoi_features = torch.cat(
            (demo_qoi_features, quest_qoi_features), dim=1
        )  # (1, demo_num + 1, qoi_dim, qoi_h, qoi_w)
        return IconData(
            cond_features=cond_features,
            qoi_features=qoi_features,
        )

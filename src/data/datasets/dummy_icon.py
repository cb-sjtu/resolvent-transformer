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

    def __init__(
        self,
        demo_num: int,
        demo_cond_dim: int,
        demo_qoi_dim: int,
        quest_cond_dim: int,
        quest_qoi_dim: int,
        demo_cond_size: tuple[int, int],
        demo_qoi_size: tuple[int, int],
        quest_cond_size: tuple[int, int],
        quest_qoi_size: tuple[int, int],
    ):
        self.demo_num = demo_num
        self.demo_cond_dim = demo_cond_dim
        self.demo_qoi_dim = demo_qoi_dim
        self.quest_cond_dim = quest_cond_dim
        self.quest_qoi_dim = quest_qoi_dim
        self.demo_cond_size = demo_cond_size
        self.demo_qoi_size = demo_qoi_size
        self.quest_cond_size = quest_cond_size
        self.quest_qoi_size = quest_qoi_size

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        demo_cond_features = torch.randn(
            1, self.demo_num, self.demo_cond_dim, self.demo_cond_size[0], self.demo_cond_size[1]
        )
        demo_qoi_features = torch.randn(
            1, self.demo_num, self.demo_qoi_dim, self.demo_qoi_size[0], self.demo_qoi_size[1]
        )
        quest_cond_features = torch.randn(1, 1, self.quest_cond_dim, self.quest_cond_size[0], self.quest_cond_size[1])
        quest_qoi_features = torch.randn(1, 1, self.quest_qoi_dim, self.quest_qoi_size[0], self.quest_qoi_size[1])

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

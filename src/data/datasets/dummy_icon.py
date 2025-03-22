from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

import src.data.data_utils as du


@dataclass
class IconData(du.DataBase):
    demo_cond_features: torch.Tensor = None
    demo_qoi_features: torch.Tensor = None
    quest_cond_features: torch.Tensor = None
    quest_qoi_features: torch.Tensor = None


class IconDataset(Dataset):
    def __init__(self, cfg: DictConfig = None):
        self.cfg = cfg

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        demo_num = self.cfg.get("demo_num")
        demo_cond_dim = self.cfg.get("demo_cond_dim")
        demo_qoi_dim = self.cfg.get("demo_qoi_dim")
        quest_cond_dim = self.cfg.get("quest_cond_dim")
        quest_qoi_dim = self.cfg.get("quest_qoi_dim")
        demo_cond_size = self.cfg.get("demo_cond_size")
        demo_qoi_size = self.cfg.get("demo_qoi_size")
        quest_cond_size = self.cfg.get("quest_cond_size")
        quest_qoi_size = self.cfg.get("quest_qoi_size")

        demo_cond_features = torch.randn(1, demo_num, demo_cond_dim, demo_cond_size[0], demo_cond_size[1])
        demo_qoi_features = torch.randn(1, demo_num, demo_qoi_dim, demo_qoi_size[0], demo_qoi_size[1])
        quest_cond_features = torch.randn(1, 1, quest_cond_dim, quest_cond_size[0], quest_cond_size[1])
        quest_qoi_features = torch.randn(1, 1, quest_qoi_dim, quest_qoi_size[0], quest_qoi_size[1])
        return IconData(
            demo_cond_features=demo_cond_features,
            demo_qoi_features=demo_qoi_features,
            quest_cond_features=quest_cond_features,
            quest_qoi_features=quest_qoi_features,
        )

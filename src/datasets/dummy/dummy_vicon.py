import torch
from torch.utils.data import Dataset

from src.datasets.data_utils import BaseLabelData, ViconData


class DummyViconDataset(Dataset):
    def __init__(self, demo_num: int, cond_shape: tuple[int, int, int], qoi_shape: tuple[int, int, int]):
        self.demo_num = demo_num
        self.cond_shape = cond_shape  # (cond_dim, cond_h, cond_w)
        self.qoi_shape = qoi_shape  # (qoi_dim, qoi_h, qoi_w)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        demo_cond = torch.randn(1, self.demo_num, self.cond_shape[0], self.cond_shape[1], self.cond_shape[2])
        demo_qoi = torch.randn(1, self.demo_num, self.qoi_shape[0], self.qoi_shape[1], self.qoi_shape[2])
        quest_cond = torch.randn(1, 1, self.cond_shape[0], self.cond_shape[1], self.cond_shape[2])
        quest_qoi = torch.randn(1, 1, self.qoi_shape[0], self.qoi_shape[1], self.qoi_shape[2])

        data = ViconData(
            description=[f"data from {self.__class__.__name__}, idx: {idx}, random state: {torch.randn(1).item()}"],
            demo_cond=demo_cond,
            demo_qoi=demo_qoi,
            quest_cond=quest_cond,
        )

        label = BaseLabelData(description=[f"label from {self.__class__.__name__}"], label=quest_qoi)

        return {"data": data, "label": label}

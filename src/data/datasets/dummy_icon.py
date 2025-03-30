import torch
from torch.utils.data import Dataset

from src.data.data_utils import IconData


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

        data = IconData(
            demo_cond_features=demo_cond_features,
            demo_qoi_features=demo_qoi_features,
            quest_cond_features=quest_cond_features,
        )

        label = quest_qoi_features

        return {"data": data, "label": label}

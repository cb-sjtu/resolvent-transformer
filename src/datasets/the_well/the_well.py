# 文件: src/datasets/thewell_active_matter/thewell_active_matter.py

import numpy as np
import torch
from the_well.data import WellDataset
from torch.utils.data import Dataset


class TheWellActiveMatterDataset(Dataset):
    def __init__(self, well_base_path, split="train", norm_stats=None, ex_num=5, c=3):
        self.dataset = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name="active_matter",
            well_split_name=split,
        )
        self.mean = (
            torch.tensor(norm_stats["mean"]).view(1, c, 1, 1) if norm_stats else None
        )
        self.std = (
            torch.tensor(norm_stats["std"]).view(1, c, 1, 1) if norm_stats else None
        )
        self.ex_num = ex_num
        self.c = c

    def __len__(self):
        return len(self.dataset) - self.ex_num - 1

    def __getitem__(self, idx):
        description = f"dataset: {self.__class__.__name__}, idx: {idx}"

        examples_f = []
        examples_g = []
        for i in range(self.ex_num):
            f = self.dataset[idx + i]["input"][: self.c]
            g = self.dataset[idx + i]["output"]
            examples_f.append(f)
            examples_g.append(g.unsqueeze(0))

        qn_f = self.dataset[idx + self.ex_num]["input"][: self.c]
        qn_g = self.dataset[idx + self.ex_num]["output"]

        examples_f = torch.stack(examples_f, dim=0).unsqueeze(0)  # (1, ex_num, c, H, W)
        examples_g = torch.stack(examples_g, dim=0).unsqueeze(0)  # (1, ex_num, 1, H, W)
        qn_f = qn_f.unsqueeze(0).unsqueeze(0)  # (1, 1, c, H, W)
        qn_g = qn_g.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        if self.mean is not None and self.std is not None:
            examples_f = (examples_f - self.mean) / self.std
            qn_f = (qn_f - self.mean) / self.std

        data = {"ex_f": examples_f, "ex_g": examples_g, "qn_f": qn_f}
        label = qn_g

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label,
        }


class TheWellActiveMatterDataset_pro(Dataset):
    def __init__(self, well_base_path, split="train", norm_stats=None, ex_num=5):
        self.dataset = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name="active_matter",
            well_split_name=split,
        )
        # 假设输入 shape: (T_in, Lx, Ly, F)
        sample = self.dataset[0]
        self.field_dim = sample["input_fields"].shape[-1]  # 通道数 F
        self.mean = (
            torch.tensor(norm_stats["mean"]).view(1, 1, 1, 1) if norm_stats else None
        )
        self.std = (
            torch.tensor(norm_stats["std"]).view(1, 1, 1, 1) if norm_stats else None
        )
        self.ex_num = ex_num

    def __len__(self):
        return (len(self.dataset) - self.ex_num - 1) * self.field_dim

    def __getitem__(self, idx):
        # 计算实际样本索引和通道索引
        true_idx = idx // self.field_dim
        channel_idx = idx % self.field_dim

        description = f"dataset: {self.__class__.__name__}, idx: {true_idx}, channel: {channel_idx}"

        examples_f = []
        examples_g = []
        for i in range(self.ex_num):
            f = self.dataset[true_idx + i]["input_fields"][
                :, :, :, channel_idx
            ]  # (T_in, Lx, Ly)
            g = self.dataset[true_idx + i]["output_fields"][
                :, :, :, channel_idx
            ]  # (T_out, Lx, Ly)
            examples_f.append(f)
            examples_g.append(g)

        qn_f = self.dataset[true_idx + self.ex_num]["input_fields"][
            :, :, :, channel_idx
        ]
        qn_g = self.dataset[true_idx + self.ex_num]["output_fields"][
            :, :, :, channel_idx
        ]

        # 转成统一格式 (1, ex_num, T or c, H, W)
        examples_f = torch.stack(examples_f, dim=0).unsqueeze(0)
        examples_g = torch.stack(examples_g, dim=0).unsqueeze(0)
        qn_f = qn_f.unsqueeze(0).unsqueeze(0)
        qn_g = qn_g.unsqueeze(0).unsqueeze(0)

        if self.mean is not None and self.std is not None:
            examples_f = (examples_f - self.mean) / self.std
            qn_f = (qn_f - self.mean) / self.std

        data = {
            "ex_f": examples_f,  # (1, ex_num, 1, T, H, W)
            "ex_g": examples_g,  # (1, ex_num, 1, T_out, H, W)
            "qn_f": qn_f,  # (1, 1, 1, T_in, H, W)
        }
        label = qn_g  # (1, 1, T_out, H, W)

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label,
        }

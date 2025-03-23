import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.components.transformer import get_transformer
from src.models.components.vicon_utils import build_alternating_block_lowtri_mask, depatchify, patchify


class Vicon(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.demo_num = cfg["demo_num"] + 1  # 1 for quest

        self.pre_proj = nn.Linear(
            in_features=cfg["transformer"]["dim_channel"] * cfg["patch_resolution"] ** 2,
            out_features=cfg["transformer"]["dim_token"],
        )
        self.post_proj = nn.Linear(
            in_features=cfg["transformer"]["dim_token"],
            out_features=cfg["transformer"]["dim_channel"] * cfg["patch_resolution"] ** 2,
        )

        self.patch_pos_encoding = nn.Parameter(
            torch.randn(cfg["patch_num_in"] * cfg["patch_num_in"], cfg["transformer"]["dim_token"])
        )
        self.func_pos_encoding = nn.Parameter(torch.randn(self.demo_num * 2, cfg["transformer"]["dim_token"]))

        self.transformer = get_transformer(cfg["transformer"], mode="encoder")

        mask = (
            1
            - build_alternating_block_lowtri_mask(
                self.demo_num, cfg["patch_num_in"] * cfg["patch_num_in"], cfg["patch_num_out"] * cfg["patch_num_out"]
            )
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        cond_features = x.cond_features
        qoi_features = x.qoi_features

        p = self.cfg["patch_num_in"]
        d = self.cfg["transformer"]["dim_token"]

        # Prepare the pairs (cond, qoi)
        x = torch.cat(
            (cond_features[:, :, None, :, :], qoi_features[:, :, None, :, :]), dim=2
        )  # (bs, pairs, 2, c, h, w)
        bs, pairs, _, c, h, w = x.shape

        feature = x.view(-1, *x.shape[-3:])  # (bs * pairs * 2, c, h, w)
        c, ph, pw = feature.shape[-3:]
        h = ph // p
        w = pw // p
        feature = patchify(feature, patch_num=p)  # (bs * pairs * 2, p * p, c * h * w)

        feature = self.pre_proj(feature)  # (bs * pairs * 2, p * p, d_model)

        feature = feature + self.patch_pos_encoding  # (bs * pairs * 2, p * p, d_model)
        feature = feature.view(bs, -1, p * p, d)  # (bs, pairs * 2, p * p, d_model)

        func_pos_encoding = self.func_pos_encoding.view(1, -1, 1, d)  # (1, cfg["demo_num"] * 2, 1, d_model)
        func_pos_encoding = func_pos_encoding[:, : pairs * 2, :, :]  # (1, pairs * 2, 1, d_model)
        feature = feature + func_pos_encoding  # (bs, pairs * 2, p * p, d_model)
        feature = feature.view(bs, -1, d)  # (bs, pairs * 2 * p * p, d_model)

        mask = self.mask[: pairs * 2 * p * p, : pairs * 2 * p * p]  # (pairs * 2 * p * p, pairs * 2 * p * p)
        feature = self.transformer(feature, mask=mask)  # (bs, pairs * 2 * p * p, d_model)
        feature = feature.view(bs, pairs, 2, p * p, d)  # (bs, pairs, 2, p * p, d_model)
        feature = feature[:, :, 0, :, :]  # (bs, pairs, p * p, d_model) the predicted QoI

        feature = self.post_proj(feature)  # (bs, pairs, p * p, c * h * w)

        feature = feature.view(bs * pairs, *feature.shape[-2:])  # (bs * pairs, p * p, c * h * w)
        feature = depatchify(feature, patch_num=p, c=c, h=h, w=w)  # (bs * pairs, c, ph, pw)
        feature = feature.view(bs, pairs, *feature.shape[-3:])  # (bs, pairs, c, ph, pw)

        return feature

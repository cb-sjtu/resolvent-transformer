import torch
import torch.nn as nn

from .vicon_utils import build_alternating_block_lowtri_mask


def patchify_pro(input_tensor, patch_size_h, patch_size_w):
    B, C, H, W = input_tensor.shape
    assert H % patch_size_h == 0 and W % patch_size_w == 0, "H and W must be divisible by patch sizes"
    return (
        input_tensor.unfold(2, patch_size_h, patch_size_h)
        .unfold(3, patch_size_w, patch_size_w)
        .contiguous()
        .view(B, C, -1, patch_size_h, patch_size_w)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(B, -1, C * patch_size_h * patch_size_w)
    )


def depatchify_pro(patches, num_patches_h, num_patches_w, c, patch_size_h, patch_size_w):
    """
    patches: (B, num_patches_h * num_patches_w, c * patch_size_h * patch_size_w)
    return: (B, c, num_patches_h * patch_size_h, num_patches_w * patch_size_w)
    """
    B = patches.shape[0]
    patches = patches.view(B, num_patches_h, num_patches_w, c, patch_size_h, patch_size_w)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return patches.view(B, c, num_patches_h * patch_size_h, num_patches_w * patch_size_w)


class Vicon(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        patch_resolution: tuple[int, int],
        patch_num_in: tuple[int, int],
        patch_num_out: tuple[int, int],
        ex_num,
        dim_channel,
        dim_token,
    ):
        super().__init__()

        self.patch_resolution = patch_resolution  # (patch_h, patch_w)
        self.patch_num_in = patch_num_in  # (num_patches_h, num_patches_w)
        self.patch_num_out = patch_num_out
        self.ex_num = ex_num + 1
        self.dim_channel = dim_channel
        self.dim_token = dim_token

        self.pre_proj = nn.Linear(
            in_features=self.dim_channel * self.patch_resolution[0] * self.patch_resolution[1],
            out_features=self.dim_token,
        )
        self.post_proj = nn.Linear(
            in_features=self.dim_token,
            out_features=self.dim_channel * self.patch_resolution[0] * self.patch_resolution[1],
        )

        total_patches = self.patch_num_in[0] * self.patch_num_in[1]
        self.patch_pos_encoding = nn.Parameter(torch.randn(total_patches, self.dim_token))
        self.func_pos_encoding = nn.Parameter(torch.randn(self.ex_num * 2, self.dim_token))

        self.transformer = transformer

        mask = (
            1
            - build_alternating_block_lowtri_mask(
                self.ex_num,
                self.patch_num_in[0] * self.patch_num_in[1],
                self.patch_num_out[0] * self.patch_num_out[1],
            )
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, f, g):
        p_h, p_w = self.patch_num_in
        d = self.dim_token

        x = torch.cat((f[:, :, None, :, :], g[:, :, None, :, :]), dim=2)  # (bs, pairs, 2, c, h, w)
        bs, pairs, _, c, ph, pw = x.shape

        feature = x.view(-1, *x.shape[-3:])  # (bs * pairs * 2, c, h, w)

        feature = patchify_pro(feature, patch_size_h=self.patch_resolution[0], patch_size_w=self.patch_resolution[1])
        feature = self.pre_proj(feature)

        feature = feature + self.patch_pos_encoding
        total_patches = p_h * p_w
        feature = feature.view(bs, -1, total_patches, d)

        func_pos_encoding = self.func_pos_encoding.view(1, -1, 1, d)
        func_pos_encoding = func_pos_encoding[:, : pairs * 2, :, :]
        feature = feature + func_pos_encoding
        feature = feature.view(bs, -1, d)

        mask = self.mask[: feature.shape[1], : feature.shape[1]]
        feature = self.transformer(feature, mask=mask)
        feature = feature.view(bs, pairs, 2, total_patches, d)
        feature = feature[:, :, 0, :, :]

        feature = self.post_proj(feature)
        feature = feature.view(bs * pairs, *feature.shape[-2:])
        feature = depatchify_pro(
            feature,
            num_patches_h=p_h,
            num_patches_w=p_w,
            c=c,
            patch_size_h=self.patch_resolution[0],
            patch_size_w=self.patch_resolution[1],
        )
        feature = feature.view(bs, pairs, *feature.shape[-3:])

        ex_pred = feature[:, :-1, :, :, :]
        qn_pred = feature[:, -1:, :, :, :]
        return {"ex_pred": ex_pred, "qn_pred": qn_pred}

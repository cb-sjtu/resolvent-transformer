"""
Simple Swin Transformer for 3-plane flow sequence prediction.
基于原始SwinTransformer2DWithMerging，只是将通道数从3改为12
"""

import einops
import torch
import torch.nn as nn

from .swin_transformer import (
    PatchEmbed2D,
    PatchExpand2D,
    PatchMerging2D,
    SwinTransformerBlock2D,
    TemporalAttention,
)

# 简化版本：直接使用原始SwinTransformer2DWithMerging的逻辑，只是将通道数改为12


class SwinTransformer3PlaneSimple(nn.Module):
    """
    Simple Swin Transformer for 3-plane flow prediction.
    基于原始SwinTransformer2DWithMerging，只是将通道数从3改为12
    """

    def __init__(
        self,
        input_shape=None,
        sequence_length=5,
        prediction_horizon=1,
        num_channels=12,  # 3 planes × 4 fields = 12 channels
        patch_size=None,
        embed_dim=96,
        depths=None,
        num_heads=8,
        window_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        final_upsample="expand_first",
        use_patch_merging=True,
        # 位置编码参数（可选）
        pos_embed_dim=64,
        channel_attn_hidden_dim=128,
        channel_attn_num_heads=8,
        **kwargs,
    ):
        super().__init__()

        # Set default values for mutable parameters
        if input_shape is None:
            input_shape = [128, 128]
        if patch_size is None:
            patch_size = [4, 4]
        if depths is None:
            depths = [2, 4, 4, 6, 4, 4, 2]  # 对称的encoder-decoder结构
        if window_size is None:
            window_size = [7, 7]

        # Validate depths parameter
        assert len(depths) % 2 == 1, (
            "depths must have odd length for symmetric encoder-decoder structure"
        )

        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # Split depths into encoder, latent, and decoder
        self.num_encoder_layers = len(depths) // 2
        self.depths_encoder = depths[: self.num_encoder_layers]
        self.depth_latent = depths[self.num_encoder_layers]
        self.depths_decoder = depths[self.num_encoder_layers + 1 :]

        self.num_layers = len(self.depths_encoder)
        self.num_layers_decoder = len(self.depths_decoder)
        self.final_upsample = final_upsample
        self.use_patch_merging = use_patch_merging

        H, W = input_shape
        self.patch_H = H // patch_size[0]
        self.patch_W = W // patch_size[1]

        # Temporal embedding - combine sequence into channels
        self.temporal_conv = nn.Conv2d(
            sequence_length * self.num_channels,
            sequence_length * self.num_channels,
            kernel_size=3,
            padding=1,
            groups=sequence_length * self.num_channels,
        )

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, sequence_length, self.num_channels, 1, 1)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Enhanced patch embedding with channel attention
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=self.num_channels,
            embed_dim=embed_dim,
            d_c=32,  # Channel projection dimension
            use_channel_attention=True,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # Temporal attention module
        self.temporal_attn = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )

        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths_encoder))
        ]
        dpr_decoder = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths_decoder))
        ]

        # Build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock2D(
                        dim=int(embed_dim * 2**i_layer),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=(0, 0)
                        if (i % 2 == 0)
                        else tuple(ws // 2 for ws in window_size),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(self.depths_encoder[:i_layer]) + i],
                        norm_layer=norm_layer,
                    )
                    for i in range(self.depths_encoder[i_layer])
                ]
            )
            self.layers.append(layer)

            # Add patch merging layer (except for the last layer)
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging2D(
                    input_resolution=(
                        self.patch_H // (2**i_layer),
                        self.patch_W // (2**i_layer),
                    ),
                    dim=int(embed_dim * 2**i_layer),
                    norm_layer=norm_layer,
                )
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(None)

        # Build decoder layers
        self.layers_decoder = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers_decoder):
            # After upsampling, dimension is halved, then doubled by concatenation
            dim_after_upsample = (
                int(embed_dim * 2 ** (self.num_encoder_layers - 1 - i_layer)) // 2
            )
            concat_linear = (
                nn.Linear(
                    2
                    * dim_after_upsample,  # Input: concatenated features (upsampled + skip)
                    dim_after_upsample,  # Output: back to upsampled dimension
                )
                if i_layer < self.num_layers_decoder - 1
                else nn.Identity()
            )
            self.concat_back_dim.append(concat_linear)

            layer = nn.ModuleList(
                [
                    SwinTransformerBlock2D(
                        dim=dim_after_upsample,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=(0, 0)
                        if (i % 2 == 0)
                        else tuple(ws // 2 for ws in window_size),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr_decoder[sum(self.depths_decoder[:i_layer]) + i],
                        norm_layer=norm_layer,
                    )
                    for i in range(self.depths_decoder[i_layer])
                ]
            )
            self.layers_decoder.append(layer)

            if i_layer < self.num_layers_decoder - 1:
                upsample = PatchExpand2D(
                    input_resolution=(
                        self.patch_H // (2 ** (self.num_encoder_layers - 1 - i_layer)),
                        self.patch_W // (2 ** (self.num_encoder_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_encoder_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
                self.upsample_layers.append(upsample)
            else:
                self.upsample_layers.append(None)

        self.norm = norm_layer(self.num_channels)

        # Final patch expanding layer
        if self.final_upsample == "expand_first":
            # Calculate final dimension after all processing
            final_dim = int(
                embed_dim * 2 ** (self.num_encoder_layers - self.num_layers_decoder)
            )
            self.up = PatchExpand2D(
                input_resolution=(self.patch_H, self.patch_W),
                dim=final_dim,
                dim_scale=self.patch_size[0],  # Expand back to original resolution
                norm_layer=norm_layer,
            )
            self.output = nn.Conv2d(
                in_channels=final_dim // self.patch_size[0],
                out_channels=self.prediction_horizon * self.num_channels,
                kernel_size=1,
                bias=False,
            )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) input sequence where C = 12
        Returns:
            output: (B, T_pred, C, H, W) predicted sequence
        """
        # Handle flattened input
        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT // (self.sequence_length * self.num_channels)
            T = self.sequence_length
            x = einops.rearrange(
                x, "(b t c) h w -> b t c h w", b=B, t=T, c=self.num_channels
            )
        else:
            B, T, C, H, W = x.shape

        assert self.sequence_length == T, (
            f"Expected sequence length {self.sequence_length}, got {T}"
        )
        assert self.num_channels == C, f"Expected {self.num_channels} channels, got {C}"
        assert tuple(self.input_shape) == (H, W), (
            f"Expected shape {self.input_shape}, got {(H, W)}"
        )

        # Add temporal position embedding
        x = x + self.temporal_pos_embed  # (B, T, C=12, H, W)

        # Reshape for temporal conv: (B, T*C, H, W)
        x = einops.rearrange(x, "b t c h w -> b (t c) h w")
        x = self.temporal_conv(x)

        # Enhanced patch embedding with channel attention
        # Input: (B, T*C, H, W) -> Output: (B, T, N, embed_dim)
        x, (patch_H, patch_W) = self.patch_embed(
            x, sequence_length=self.sequence_length
        )
        x = self.pos_drop(x)

        # Apply temporal attention across T dimension
        # Input: (B, T, N, embed_dim) -> Output: (B, T, N, embed_dim)
        x = self.temporal_attn(x)

        # Reshape for spatial processing: (B, T, N, embed_dim) -> (B*T, N, embed_dim)
        x = einops.rearrange(x, "b t n c -> (b t) n c")

        # Store encoder features for skip connections
        x_downsample = []

        # Encoder
        for i_layer, (layer, downsample) in enumerate(
            zip(self.layers, self.downsample_layers, strict=False)
        ):
            # Store current features for skip connection
            x_downsample.append(x)

            # Apply transformer blocks
            current_resolution = (
                self.patch_H // (2**i_layer),
                self.patch_W // (2**i_layer),
            )
            for block in layer:
                x = block(x, current_resolution[0], current_resolution[1])

            # Patch merging (downsampling)
            if downsample is not None:
                x = downsample(x)

        # Decoder with skip connections
        for i_layer, (layer_up, layer, concat_back_dim) in enumerate(
            zip(
                self.upsample_layers,
                self.layers_decoder,
                self.concat_back_dim,
                strict=False,
            )
        ):
            # Apply transformer blocks first
            current_resolution = (
                self.patch_H // (2 ** (self.num_encoder_layers - 1 - i_layer)),
                self.patch_W // (2 ** (self.num_encoder_layers - 1 - i_layer)),
            )
            for block in layer:
                x = block(x, current_resolution[0], current_resolution[1])

            # Upsample (except for last decoder layer)
            if i_layer < self.num_layers_decoder - 1:
                x = layer_up(x)
                # Concatenate with encoder features (skip connection)
                skip_idx = (
                    self.num_encoder_layers - 2 - i_layer
                )  # Correct skip connection index
                if skip_idx >= 0:
                    x = torch.cat([x, x_downsample[skip_idx]], -1)
                    x = concat_back_dim(x)

        x = self.norm(x)

        # Final upsampling and output
        if self.final_upsample == "expand_first":
            x = self.up(x)
            # Calculate actual spatial dimensions after final upsampling
            # PatchExpand2D with dim_scale=patch_size[0] expands by patch_size[0] in each spatial dimension
            final_H = self.patch_H * self.patch_size[0]
            final_W = self.patch_W * self.patch_size[1]

            # x is currently (B*T, expanded_patches, reduced_dim)
            # We need to reshape it back to (B, final_H, final_W, final_channels)
            BT = x.shape[0]
            assert BT == B * T, f"Expected BT={B * T}, got {BT}"

            # Reshape (B*T, expanded_patches, reduced_dim) -> (B*T, final_H, final_W, reduced_dim)
            x = x.view(BT, final_H, final_W, -1)
            # Now merge time dimension: (B*T, final_H, final_W, reduced_dim) -> (B, final_H, final_W, T*reduced_dim)
            x = x.view(B, T, final_H, final_W, -1)
            x = einops.rearrange(x, "b t h w c -> b h w (t c)")

            x = x.permute(0, 3, 1, 2)  # (B, T*reduced_dim, H, W)
            x = self.output(x)  # (B, prediction_horizon * num_channels, H, W)

            # Reshape output to (B, prediction_horizon, num_channels, H, W)
            x = x.view(B, self.prediction_horizon, self.num_channels, final_H, final_W)

        if self.prediction_horizon == 1:
            x = x.squeeze(1)  # (B, C, H, W)

        return x


def SwinTransformer3PlaneSimpleAuto(**kwargs):
    """Factory function for simple 3-plane Swin Transformer."""
    return SwinTransformer3PlaneSimple(**kwargs)

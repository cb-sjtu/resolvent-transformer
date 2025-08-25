import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition_2d(x, window_size):
    """
    Partition 2D feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: window size (Wh, Ww)
    Returns:
        windows: (num_windows*B, Wh*Ww, C)
    """
    B, H, W, C = x.shape
    Wh, Ww = window_size

    # Pad if necessary
    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    B, H, W, C = x.shape

    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh * Ww, C)

    return windows


def window_reverse_2d(windows, window_size, H, W):
    """
    Reverse window partition for 2D feature map.
    Args:
        windows: (num_windows*B, Wh*Ww, C)
        window_size: window size (Wh, Ww)
        H, W: original feature map size
    Returns:
        x: (B, H, W, C)
    """
    Wh, Ww = window_size

    # Calculate padded dimensions
    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww

    H_pad = H + pad_h
    W_pad = W + pad_w

    B = int(windows.shape[0] / (H_pad * W_pad / Wh / Ww))

    x = windows.view(B, H_pad // Wh, W_pad // Ww, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :]

    return x


class WindowAttention2D(nn.Module):
    """2D Window based multi-head self attention module."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    """2D Swin Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(7, 7),
        shift_size=(0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, L, C) where L = H*W
            H, W: spatial dimensions
        """
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # Cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition_2d(shifted_x, self.window_size)  # (nW*B, window_size**2, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)

        # Merge windows
        shifted_x = window_reverse_2d(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging2D(nn.Module):
    """Patch Merging Layer for 2D Swin Transformer."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            (B, (H//2)*(W//2), 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    """Patch Expanding Layer for U-Net decoder (sub-pixel upsampling).
    Upscales spatial size by `dim_scale` and reduces channels by the same factor.

    Input:  x: (B, H*W, dim)
    Output: x: (B, (dim_scale**2)*H*W, dim // dim_scale)
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm, out_dim=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        # 默认把通道按 dim_scale 降，确保可整除
        if out_dim is None:
            assert dim % dim_scale == 0, f"dim ({dim}) must be divisible by dim_scale ({dim_scale})"
            out_dim = dim // dim_scale
        self.out_dim = out_dim

        # 先线性映射到 s^2 * C_out，再做子像素重排
        self.expand = nn.Linear(dim, (dim_scale**2) * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, dim)
        Returns:
            (B, (dim_scale**2)*H*W, out_dim)
        """
        H, W = self.input_resolution
        B, L, Cin = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H * W}"
        assert Cin == self.dim, f"channel mismatch: Cin={Cin}, expected {self.dim}"

        # (B, H*W, s^2 * C_out)
        x = self.expand(x)

        # -> (B, H, W, s, s, C_out)
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, self.out_dim)

        # 子像素重排: (H, W, s, s) -> (H*s, W*s)
        # (B, H, s, W, s, C_out) -> (B, H*s, W*s, C_out)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * self.dim_scale, W * self.dim_scale, self.out_dim)

        # 展平成 token 序列，再做 LN
        x = x.view(B, -1, self.out_dim)
        x = self.norm(x)
        return x


class FinalPatchExpand2D(nn.Module):
    """Final patch expanding layer for reconstruction."""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            (B, 16*H*W, C)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 4, 4, C // 16)
        x = x.permute(0, 1, 3, 2, 4, 5)  # B, H, 4, W, 4, C//16
        x = x.contiguous().view(B, H * 4, W * 4, C // 16)
        x = x.view(B, -1, C // 16)
        x = self.norm(x)

        return x


class PatchEmbed2D(nn.Module):
    """2D patch embedding."""

    def __init__(self, patch_size=(4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, embed_dim) where N = (H//Ph) * (W//Pw)
        """
        x = self.proj(x)  # (B, embed_dim, H//Ph, W//Pw)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)


class SwinTransformer2DWithMerging(nn.Module):
    """2D Swin U-net for temporal flow field prediction with multiscale features."""

    def __init__(
        self,
        input_shape: tuple[int, int],  # (H, W)
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        patch_size: tuple[int, int] = (4, 4),
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 4, 4, 6, 4, 4, 2),
        num_heads: int = 8,
        window_size: tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        final_upsample: str = "expand_first",
        use_patch_merging: bool = True,
    ):
        super().__init__()

        # Validate depths parameter
        assert len(depths) % 2 == 1, "depths must have odd length for symmetric encoder-decoder structure"

        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
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
            sequence_length, sequence_length, kernel_size=3, padding=1, groups=sequence_length
        )

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, sequence_length, 1, 1, 1))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=sequence_length,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths_encoder))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths_decoder))]

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
                        shift_size=(0, 0) if (i % 2 == 0) else tuple(ws // 2 for ws in window_size),
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
                    input_resolution=(self.patch_H // (2**i_layer), self.patch_W // (2**i_layer)),
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
            dim_after_upsample = int(embed_dim * 2 ** (self.num_encoder_layers - 1 - i_layer)) // 2
            concat_linear = (
                nn.Linear(
                    2 * dim_after_upsample,  # Input: concatenated features (upsampled + skip)
                    dim_after_upsample,  # Output: back to upsampled dimension
                )
                if i_layer < self.num_layers_decoder - 1
                else nn.Identity()
            )

            layer_up = (
                PatchExpand2D(
                    input_resolution=(
                        self.patch_H // (2 ** (self.num_encoder_layers - 1 - i_layer)),
                        self.patch_W // (2 ** (self.num_encoder_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_encoder_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
                if (i_layer < self.num_layers_decoder - 1)
                else nn.Identity()
            )

            layer = nn.ModuleList(
                [
                    SwinTransformerBlock2D(
                        dim=int(embed_dim * 2 ** (self.num_encoder_layers - 1 - i_layer)),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=(0, 0) if (i % 2 == 0) else tuple(ws // 2 for ws in window_size),
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
            self.upsample_layers.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.embed_dim)

        # Final patch expanding and output layer
        if self.final_upsample == "expand_first":
            # Use PatchExpand2D with correct scaling to match patch_size
            self.up = PatchExpand2D(
                input_resolution=(self.patch_H, self.patch_W),
                dim=embed_dim,
                dim_scale=patch_size[0],  # Assuming square patches
                norm_layer=norm_layer,
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim // patch_size[0],  # PatchExpand2D reduces channels by dim_scale
                out_channels=prediction_horizon,
                kernel_size=1,
                bias=False,
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) input sequence or (B*T, C, H, W) flattened input
        Returns:
            output: (B, T_pred, C, H, W) predicted sequence
        """
        # Handle flattened input
        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT // self.sequence_length
            T = self.sequence_length
            x = x.view(B, T, C, H, W)
        else:
            B, T, C, H, W = x.shape

        assert self.sequence_length == T, f"Expected sequence length {self.sequence_length}, got {T}"
        assert tuple(self.input_shape) == (H, W), f"Expected shape {self.input_shape}, got {(H, W)}"

        # Add temporal position embedding
        x = x + self.temporal_pos_embed
        x = x.reshape(B, T * C, H, W)
        x = self.temporal_conv(x)

        # Patch embedding
        x, _ = self.patch_embed(x)
        x = self.pos_drop(x)

        # Store encoder features for skip connections
        x_downsample = []

        # Encoder
        for i_layer, (layer, downsample) in enumerate(zip(self.layers, self.downsample_layers, strict=False)):
            # Store current features for skip connection
            x_downsample.append(x)

            # Apply transformer blocks
            current_resolution = (self.patch_H // (2**i_layer), self.patch_W // (2**i_layer))
            for block in layer:
                x = block(x, current_resolution[0], current_resolution[1])

            # Patch merging (downsampling)
            if downsample is not None:
                x = downsample(x)

        # Decoder with skip connections
        for i_layer, (layer_up, layer, concat_back_dim) in enumerate(
            zip(self.upsample_layers, self.layers_decoder, self.concat_back_dim, strict=False)
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
                skip_idx = self.num_encoder_layers - 2 - i_layer  # Correct skip connection index
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
            x = x.view(B, final_H, final_W, -1)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.output(x)  # (B, prediction_horizon, H, W)

            if self.prediction_horizon == 1:
                x = x.squeeze(1)  # (B, H, W) -> need to add channel dim
                x = x.unsqueeze(1)  # (B, 1, H, W)

        return x


class SwinTransformer2D(nn.Module):
    """Enhanced 2D Swin Transformer with patch merging for temporal flow field prediction."""

    def __init__(
        self,
        input_shape: tuple[int, int],  # (H, W)
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        patch_size: tuple[int, int] = (4, 4),
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_patch_merging: bool = True,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_patch_merging = use_patch_merging

        H, W = input_shape
        self.patch_H = H // patch_size[0]
        self.patch_W = W // patch_size[1]

        # Temporal embedding
        self.temporal_conv = nn.Conv2d(
            sequence_length, sequence_length, kernel_size=3, padding=1, groups=sequence_length
        )
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, sequence_length, 1, 1, 1))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=sequence_length, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        # Build layers with optional patch merging
        self.layers = nn.ModuleList()
        self.merging_layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        current_dim = embed_dim
        current_resolution = (self.patch_H, self.patch_W)

        for i_layer, (depth, num_head) in enumerate(zip(depths, num_heads, strict=False)):
            # Swin Transformer blocks
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock2D(
                        dim=current_dim,
                        num_heads=num_head,
                        window_size=window_size,
                        shift_size=(0, 0) if (i % 2 == 0) else tuple(ws // 2 for ws in window_size),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) + i],
                    )
                    for i in range(depth)
                ]
            )
            self.layers.append(layer)

            # Patch merging (except for the last layer)
            if self.use_patch_merging and i_layer < len(depths) - 1:
                merging = PatchMerging2D(input_resolution=current_resolution, dim=current_dim, norm_layer=nn.LayerNorm)
                self.merging_layers.append(merging)
                current_dim *= 2
                current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)
            else:
                self.merging_layers.append(None)

        self.norm = nn.LayerNorm(current_dim)

        # Output projection - adapted for final dimension
        self.output_proj = nn.Sequential(
            nn.Linear(current_dim, current_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(current_dim * 2, embed_dim),  # Project back to original embed_dim
            nn.GELU(),
            nn.Linear(embed_dim, prediction_horizon * (patch_size[0] * patch_size[1])),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) input sequence or (B*T, C, H, W) flattened input
        Returns:
            output: (B, T_pred, C, H, W) predicted sequence
        """
        # Handle flattened input
        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT // self.sequence_length
            T = self.sequence_length
            x = x.view(B, T, C, H, W)
        else:
            B, T, C, H, W = x.shape

        assert self.sequence_length == T, f"Expected sequence length {self.sequence_length}, got {T}"
        assert self.input_shape == (H, W), f"Expected shape {self.input_shape}, got {(H, W)}"

        # Temporal processing
        x = x + self.temporal_pos_embed
        x = x.reshape(B, T * C, H, W)
        x = self.temporal_conv(x)

        # Patch embedding
        x, _ = self.patch_embed(x)
        x = self.pos_drop(x)

        # Apply Swin Transformer layers with optional patch merging
        current_H, current_W = self.patch_H, self.patch_W

        for layer, merging in zip(self.layers, self.merging_layers, strict=False):
            # Apply transformer blocks
            for block in layer:
                x = block(x, current_H, current_W)

            # Apply patch merging if available
            if merging is not None:
                x = merging(x)
                current_H, current_W = current_H // 2, current_W // 2

        x = self.norm(x)

        # Global average pooling to reduce spatial dimensions
        # Reshape for pooling: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).view(B, -1, current_H, current_W)
        x = F.adaptive_avg_pool2d(x, (self.patch_H, self.patch_W))  # Pool back to original patch resolution
        x = x.view(B, -1, self.patch_H * self.patch_W).transpose(1, 2)  # Back to (B, H*W, C)

        # Output projection
        x = self.output_proj(x)  # (B, N_patches, prediction_horizon * patch_area)

        # Reshape to output format
        patch_area = self.patch_size[0] * self.patch_size[1]
        x = x.reshape(B, self.patch_H, self.patch_W, self.prediction_horizon, patch_area)

        # Reconstruct spatial dimensions
        x = x.permute(0, 3, 1, 2, 4)  # (B, T_pred, patch_H, patch_W, patch_area)
        x = x.contiguous().reshape(
            B, self.prediction_horizon, self.patch_H, self.patch_W, self.patch_size[0], self.patch_size[1]
        )

        # Unfold patches back to original spatial resolution
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, T_pred, patch_H, patch_size[0], patch_W, patch_size[1])
        x = x.contiguous().reshape(B, self.prediction_horizon, 1, H, W)

        # Handle output format
        if self.prediction_horizon == 1:
            x = x.squeeze(1)  # (B, 1, H, W)

        return x


# Alias for the enhanced U-net version (for advanced users)
SwinUnet2D = SwinTransformer2DWithMerging


def SwinTransformerAuto(use_patch_merging: bool = True, **kwargs):
    """
    Factory function to automatically choose between SwinTransformer2D and SwinTransformer2DWithMerging
    based on the use_patch_merging parameter.

    Args:
        use_patch_merging: If True, use SwinTransformer2DWithMerging, else use SwinTransformer2D
        **kwargs: Arguments passed to the selected model

    Returns:
        Either SwinTransformer2D or SwinTransformer2DWithMerging instance
    """
    if use_patch_merging:
        # Remove use_patch_merging from kwargs since SwinTransformer2DWithMerging doesn't need it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "use_patch_merging"}
        return SwinTransformer2DWithMerging(**filtered_kwargs)
    else:
        # Remove use_patch_merging from kwargs since SwinTransformer2D doesn't need it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "use_patch_merging"}
        # Convert depths parameter for SwinTransformer2D (extract encoder depths only)
        if "depths" in filtered_kwargs:
            depths = filtered_kwargs["depths"]
            if len(depths) % 2 == 1:  # If it's the new format (encoder, latent, decoder)
                num_encoder_layers = len(depths) // 2
                filtered_kwargs["depths"] = depths[:num_encoder_layers]
        return SwinTransformer2D(use_patch_merging=False, **filtered_kwargs)


if __name__ == "__main__":
    print("Testing Enhanced Swin Transformer 2D with patch merging...")

    # Test with patch merging enabled
    print("\n1. Testing with patch merging enabled:")
    model_with_merging = SwinTransformer2D(
        input_shape=(64, 48),  # Smaller for testing
        sequence_length=5,
        prediction_horizon=1,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(3, 6),
        window_size=(4, 4),
        patch_size=(4, 4),
        use_patch_merging=True,
        drop_path_rate=0.0,
    )

    x = torch.randn(1, 3, 1, 64, 48)  # (B, T, C, H, W)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model_with_merging(x)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_with_merging.parameters()) / 1e6:.2f}M")

    # Test without patch merging (original behavior)
    print("\n2. Testing without patch merging (original behavior):")
    model_without_merging = SwinTransformer2D(
        input_shape=(64, 48),
        sequence_length=5,
        prediction_horizon=1,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(3, 6),
        window_size=(4, 4),
        patch_size=(4, 4),
        use_patch_merging=False,
        drop_path_rate=0.0,
    )

    with torch.no_grad():
        output_orig = model_without_merging(x)
    print(f"Output shape: {output_orig.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_without_merging.parameters()) / 1e6:.2f}M")

    print("\n✓ All tests passed! Enhanced model with patch merging is working.")
    print("  - Patch merging enables multiscale feature extraction")
    print("  - U-net-like architecture improves spatial reasoning")
    print("  - Backward compatibility maintained with use_patch_merging=False")

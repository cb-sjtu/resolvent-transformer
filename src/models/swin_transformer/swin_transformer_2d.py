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


class PatchEmbed2D(nn.Module):
    """2D patch embedding."""

    def __init__(self, patch_size=(4, 4), in_chans=1, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

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
        x = self.norm(x)
        return x, (H, W)


class SwinTransformer2D(nn.Module):
    """2D Swin Transformer for temporal flow field prediction."""

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
    ):
        super().__init__()

        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        H, W = input_shape
        self.patch_H = H // patch_size[0]
        self.patch_W = W // patch_size[1]

        # Temporal embedding - combine sequence into channels
        # Input will be (B, T*C, H, W) where C=1, so T*C = sequence_length
        self.temporal_conv = nn.Conv2d(
            sequence_length, sequence_length, kernel_size=3, padding=1, groups=sequence_length
        )

        # Temporal position embedding for sequence information
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, sequence_length, 1, 1, 1))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=sequence_length, embed_dim=embed_dim)

        # Relative position embedding - will be handled by attention blocks
        # Remove absolute positional embedding
        self.pos_drop = nn.Dropout(drop_rate)

        # Build layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i_layer, (depth, num_head) in enumerate(zip(depths, num_heads, strict=False)):
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock2D(
                        dim=embed_dim,
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

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 2, prediction_horizon * (patch_size[0] * patch_size[1])),
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
            x: (B, T, C, H, W) input sequence
        Returns:
            output: (B, T_pred, C, H, W) predicted sequence
        """
        B, T, C, H, W = x.shape
        assert self.sequence_length == T, f"Expected sequence length {self.sequence_length}, got {T}"
        assert self.input_shape == (H, W), f"Expected shape {self.input_shape}, got {(H, W)}"

        # Add temporal position embedding before reshaping
        x = x + self.temporal_pos_embed  # (B, T, C, H, W)

        # Reshape to (B, T*C, H, W) for temporal processing
        # When C=1, this becomes (B, T, H, W)
        x = x.reshape(B, T * C, H, W)

        # Apply temporal convolution
        x = self.temporal_conv(x)

        # Patch embedding: (B, embed_dim, patch_H, patch_W)
        x, patch_dims = self.patch_embed(x)  # (B, N, embed_dim)

        # Apply dropout (no absolute positional embedding)
        x = self.pos_drop(x)

        # Apply Swin Transformer layers
        for layer in self.layers:
            for block in layer:
                x = block(x, self.patch_H, self.patch_W)

        x = self.norm(x)

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

        return x


if __name__ == "__main__":
    # Test the model
    model = SwinTransformer2D(
        input_shape=(128, 96),  # 2D shape (H, W)
        sequence_length=5,
        prediction_horizon=1,
        embed_dim=96,
        depths=(2, 2, 6),
        num_heads=(3, 6, 12),
        window_size=(7, 7),
        patch_size=(4, 4),
    )

    # Test input
    x = torch.randn(2, 5, 1, 128, 96)  # (B, T, C, H, W)

    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

import copy

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, ModuleList, MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, ff=True, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, **factory_kwargs
        )
        layer_norm_eps = 1e-5

        self.ff = ff

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        if ff:
            self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout2 = Dropout(dropout)
            self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        # shape = x.shape  # workaround
        # x = x.flatten(0, 1)  # workaround
        if need_weights:
            attn_out, weight = self.self_attn(
                x,
                x,
                x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out = self.self_attn(
                x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False
            )[0]
        # attn_out = attn_out.view(shape)  # workaround
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        if self.ff:
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            ff_out = self.dropout2(ff_out)
            x = self.norm2(x + ff_out)

        if need_weights:
            return x, weight
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, self_attn_layer, num_layers):
        super().__init__()

        self.layers = ModuleList([copy.deepcopy(self_attn_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        weights = []
        for i in range(self.num_layers):
            if need_weights:
                x, weight = self.layers[i](x, mask, src_key_padding_mask, need_weights=True)
                weights.append(weight)
            else:
                x = self.layers[i](x, mask, src_key_padding_mask, need_weights=False)
        if need_weights:
            return x, weights
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, ff=True, device=None, dtype=None):
        """
        Transformer decoder layer, no self attention
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.cross_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, **factory_kwargs
        )
        layer_norm_eps = 1e-5

        self.ff = ff

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        if ff:
            self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout2 = Dropout(dropout)
            self.activation = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        need_weights=False,
    ):
        x = tgt
        # shape = x.shape  # workaround
        # x = x.flatten(0, 1)  # workaround
        if need_weights:
            attn_out, weight = self.cross_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out = self.cross_attn(
                x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False
            )[0]
        # attn_out = attn_out.view(shape)  # workaround
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        if self.ff:
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            ff_out = self.dropout2(ff_out)
            x = self.norm2(x + ff_out)

        if need_weights:
            return x, weight
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()

        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        need_weights=False,
    ):
        x = tgt
        weights = []
        for i in range(self.num_layers):
            if need_weights:
                x, weight = self.layers[i](
                    x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, need_weights=True
                )
                weights.append(weight)
            else:
                x = self.layers[i](
                    x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, need_weights=False
                )
        if need_weights:
            return x, weights
        return x

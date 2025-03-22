import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # learnable parameter alpha
        self.alpha = nn.Parameter(torch.ones(1))

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        #########################################################
        # before applying softmax, multiply scores by alpha parameter
        scores = scores * self.alpha
        #########################################################

        attention = F.softmax(scores, dim=-1)

        output = torch.matmul(attention, V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        self_attn_output = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        x = self.input_embedding(x)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)

        return x


class OperatorTransformer(nn.Module):
    def __init__(
        self,
        f_input_dim=None,  # (x,y) from f
        g_input_dim=None,  # x from g
        g_output_dim=None,  # y from g
        d_model=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=1000,
        dropout=0.1,
        cfg=None,
    ):
        super().__init__()

        if cfg is not None:
            f_input_dim = int(cfg.model.f_input_dim)
            g_input_dim = int(cfg.model.g_input_dim)
            g_output_dim = int(cfg.model.g_output_dim)
            d_model = int(cfg.model.d_model)
            num_encoder_layers = int(cfg.model.num_encoder_layers)
            num_decoder_layers = int(cfg.model.num_decoder_layers)
            num_heads = int(cfg.model.num_heads)
            d_ff = int(cfg.model.d_ff)
            max_seq_len = int(cfg.model.max_seq_len)
            dropout = float(cfg.model.dropout)

        self.encoder = Encoder(
            input_dim=f_input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.decoder = Decoder(
            input_dim=g_input_dim,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(d_model, g_output_dim)

    def _create_decoder_mask(self, g_inputs):
        seq_len = g_inputs.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(g_inputs.device)
        look_ahead_mask = (1 - mask).bool()
        return look_ahead_mask.unsqueeze(0)

    def forward(self, f_samples, g_inputs, encoder_mask=None, decoder_self_mask=None, decoder_cross_mask=None):
        if decoder_self_mask is None:
            decoder_self_mask = self._create_decoder_mask(g_inputs)

        enc_output = self.encoder(f_samples, mask=None)

        dec_output = self.decoder(g_inputs, enc_output, decoder_self_mask, decoder_cross_mask)

        g_outputs = self.output_projection(dec_output)

        return g_outputs


if __name__ == "__main__":
    batch_size = 32
    f_seq_len = 100
    g_seq_len = 50
    f_input_dim = 2
    g_input_dim = 1
    g_output_dim = 1

    model = OperatorTransformer(
        f_input_dim=f_input_dim,
        g_input_dim=g_input_dim,
        g_output_dim=g_output_dim,
        d_model=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=max(f_seq_len, g_seq_len),
        dropout=0.1,
    )

    f_samples = torch.randn(batch_size, f_seq_len, f_input_dim)  # (x,y) from f
    g_inputs = torch.randn(batch_size, g_seq_len, g_input_dim)  # x from g

    g_outputs = model(f_samples, g_inputs)
    print(f"g_outputs shape: {g_outputs.shape}")

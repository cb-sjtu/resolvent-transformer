from torch import nn

from .transformer import get_transformer


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder_in_proj: nn.Linear,
        decoder_in_proj: nn.Linear,
        encoder: nn.Module,
        decoder: nn.Module,
        out_proj: nn.Linear,
    ):
        super().__init__()
        self.encoder_in_proj = encoder_in_proj
        self.decoder_in_proj = decoder_in_proj
        self.encoder = encoder
        self.decoder = decoder
        self.out_proj = out_proj

    def encoder_in_proj(self, f_input_dim, model_dim):
        self.encoder_in_proj = nn.Linear(f_input_dim, model_dim)

    def decoder_in_proj(self, g_input_dim, model_dim):
        self.decoder_in_proj = nn.Linear(g_input_dim, model_dim)

    def encoder(self, model_dim, n_heads, widening_factor, n_layers):
        self.encoder = get_transformer(model_dim, n_heads, widening_factor, n_layers, mode="encoder")

    def decoder(self, model_dim, n_heads, widening_factor, n_layers):
        self.decoder = get_transformer(model_dim, n_heads, widening_factor, n_layers, mode="decoder")

    def out_proj(self, model_dim):
        self.out_proj = nn.Linear(model_dim, 1)

    def forward(self, memory, query):
        memory = self.encoder_in_proj(memory)
        query = self.decoder_in_proj(query)
        x = self.encoder(memory)
        x = self.decoder(query, x)
        x = self.out_proj(x)
        return x

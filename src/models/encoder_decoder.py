from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder_in_proj: nn.Module,
        decoder_in_proj: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        out_proj: nn.Module,
        f_input_dim: int = 2,
        g_input_dim: int = 1,
        widening_factor: int = 4,
        dropout: float = 0.0,
        compile: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.encoder_in_proj = encoder_in_proj
        self.decoder_in_proj = decoder_in_proj
        self.encoder = encoder
        self.decoder = decoder
        self.out_proj = out_proj

        self.f_input_dim = f_input_dim
        self.g_input_dim = g_input_dim
        self.widening_factor = widening_factor
        self.dropout = dropout
        self.compile = compile

    def forward(self, memory, query):
        memory = self.encoder_in_proj(memory)
        query = self.decoder_in_proj(query)
        x = self.encoder(memory)
        x = self.decoder(query, x)
        x = self.out_proj(x)
        return x

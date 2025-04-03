from omegaconf import DictConfig
from torch import nn

from .transformer import get_transformer


class EncoderDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_in_proj = nn.Linear(cfg.model.f_input_dim, cfg.model.encoder.model_dim)
        self.decoder_in_proj = nn.Linear(cfg.model.g_input_dim, cfg.model.decoder.model_dim)
        self.encoder = get_transformer(cfg.model.encoder, mode="encoder")
        self.decoder = get_transformer(cfg.model.decoder, mode="decoder")
        self.out_proj = nn.Linear(cfg.model.decoder.model_dim, 1)

    def forward(self, memory, query):
        memory = self.encoder_in_proj(memory)
        query = self.decoder_in_proj(query)
        x = self.encoder(memory)
        x = self.decoder(query, x)
        x = self.out_proj(x)
        return x

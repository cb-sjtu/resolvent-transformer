from omegaconf import DictConfig
from torch import nn

from src.data.datasets.dummy_operator import OperatorData

from .transformer import get_transformer


class EncDecOL(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_in_proj = nn.Linear(cfg.model.f_input_dim, cfg.model.encoder.model_dim)
        self.decoder_in_proj = nn.Linear(cfg.model.g_input_dim, cfg.model.decoder.model_dim)
        self.encoder = get_transformer(cfg.model.encoder, mode="encoder")
        self.decoder = get_transformer(cfg.model.decoder, mode="decoder")
        self.out_proj = nn.Linear(cfg.model.decoder.model_dim, 1)

    def forward(self, x: OperatorData):
        f_samples = x.f_samples
        g_inputs = x.g_inputs
        f_samples = self.encoder_in_proj(f_samples)
        g_inputs = self.decoder_in_proj(g_inputs)
        x = self.encoder(f_samples)
        x = self.decoder(g_inputs, x)
        x = self.out_proj(x)
        return x

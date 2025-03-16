import torch
from einops import rearrange

# typing
from omegaconf import DictConfig
from torch import nn

from src.data import data_utils as du

from .transformer import get_transformer


def get_decoder_input_dim(cfg: DictConfig):
    return 2  # 2D principal invariants


def get_encoder_pre_projection(cfg: DictConfig):
    if cfg.loss.prompt == "mesh":
        # just a linear layer
        return nn.Linear(5, cfg.model.encoder.model_dim)
    elif cfg.loss.prompt == "equation":
        # transformer pooler
        return EquilUnitEncoder(cfg)
    else:
        raise ValueError(f"Unknown prompt: {cfg.loss.prompt}")


def get_output_dim(cfg: DictConfig):
    if cfg.loss.output == "psi":
        return 1
    elif cfg.loss.output == "grad_psi_grad_I":
        return 2
    else:
        raise ValueError(f"Unknown output: {cfg.loss.output}")


class EquilUnitEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.pool_token = nn.Parameter(torch.randn(cfg.model.preprocessor.input_dim))
        self.projector = nn.Linear(cfg.model.preprocessor.input_dim, cfg.model.preprocessor.model_dim)
        self.encoder = get_transformer(cfg.model.preprocessor, mode="encoder")

    def forward(self, x: du.DataEqn) -> torch.Tensor:
        # x: DataEqn
        mask = ~x.mask  # (bs, units, max_elements)
        # (bs, units, max_elements+1), zero for the pool token
        mask = torch.cat([torch.zeros_like(mask[:, :, -1:]), mask], dim=-1)
        A_flatten = x.A.reshape(*x.A.shape[:-2], -1)  # (bs, units, max_elements, 4)
        prompt = torch.cat([A_flatten, x.XI], dim=-1)  # (bs, units, max_elements, 4 + 2)
        bs, units, max_elements, dim = prompt.shape
        prompt = prompt.reshape(-1, prompt.shape[-2], prompt.shape[-1])  # (bs*units, max_elements, dim)
        mask = mask.reshape(-1, mask.shape[-1])  # (bs*units, max_elements+1)
        pool_token = self.pool_token.unsqueeze(0).unsqueeze(0).repeat(prompt.shape[0], 1, 1)  # (bs * units, 1, dim)
        prompt_pool = torch.cat([pool_token, prompt], dim=-2)  # (bs*units, max_elements+1, dim)
        prompt_pool = self.projector(prompt_pool)  # (bs*units, max_elements+1, model_dim)
        # mask: a True value indicates that the corresponding key value will be ignored for the purpose of attention
        prompt_pool = self.encoder(prompt_pool, src_key_padding_mask=mask)  # (bs*units, max_elements+1, model_dim)
        prompt_pool = prompt_pool[:, 0, :]  # (bs*units, model_dim)
        prompt_pool = prompt_pool.reshape(bs, units, -1)  # (bs, units, model_dim)
        return prompt_pool


class ICE_EncoderDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Encoder-Decoder model for ICON
        """
        super().__init__()

        self.encoder_pre_projection = get_encoder_pre_projection(cfg)
        self.decoder_pre_projection = nn.Linear(get_decoder_input_dim(cfg), cfg.model.decoder.model_dim)
        self.encoder = get_transformer(cfg.model.encoder, mode="encoder")
        self.decoder = get_transformer(cfg.model.decoder, mode="decoder")
        self.post_projection = nn.Linear(cfg.model.decoder.model_dim, get_output_dim(cfg))

    def forward(self, prompt: du.DataEqn, query: torch.Tensor, **kwargs):
        memory = self.encoder_pre_projection(prompt)
        memory = self.encoder(memory)
        output = self.decoder_pre_projection(query)
        output = rearrange(output, "b ... d -> b (...) d")
        output = self.decoder(output, memory)
        output = self.post_projection(output)
        output = output.view(output.shape[0], *query.shape[1:-1], output.shape[-1])
        return output  # (bs, n, output_dim)

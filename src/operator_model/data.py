from dataclasses import dataclass
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import src.data.data_utils as du

@dataclass
class OperatorData(du.DataBase):
    f_samples: torch.Tensor = None
    g_inputs: torch.Tensor = None
    g_targets: torch.Tensor = None

class OperatorDataset(Dataset):
    def __init__(self, cfg: DictConfig = None):
        self.cfg = cfg

    def __len__(self):
        return 10000   

    def __getitem__(self, idx):
        f_seq_len = 5
        g_seq_len = 2
        f_input_dim = 2
        g_input_dim = 1
        g_output_dim = 1
        
        if self.cfg is not None and "model" in self.cfg:
            f_seq_len = self.cfg.model.get("f_seq_len", f_seq_len)
            g_seq_len = self.cfg.model.get("g_seq_len", g_seq_len)
            f_input_dim = self.cfg.model.get("f_input_dim", f_input_dim)
            g_input_dim = self.cfg.model.get("g_input_dim", g_input_dim)
            g_output_dim = self.cfg.model.get("g_output_dim", g_output_dim)
        
        f_samples = torch.randn(f_seq_len, f_input_dim).unsqueeze(0) 
        g_inputs = torch.randn(g_seq_len, g_input_dim).unsqueeze(0)   
        g_targets = torch.randn(g_seq_len, g_output_dim).unsqueeze(0) 
        
        return OperatorData(
            f_samples=f_samples,
            g_inputs=g_inputs,
            g_targets=g_targets
        )

class OperatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def _get_data_params(self):
        batch_size = 32
        num_workers = 4
        pin_memory = True
        
        if self.cfg is not None and "data" in self.cfg:
            batch_size = self.cfg.data.get("batch_size", batch_size)
            num_workers = self.cfg.data.get("num_workers", num_workers)
            pin_memory = self.cfg.data.get("pin_memory", pin_memory)
            
        return batch_size, num_workers, pin_memory

    def train_dataloader(self):
        batch_size, num_workers, pin_memory = self._get_data_params()
        
        return DataLoader(
            dataset=OperatorDataset(cfg=self.cfg),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        batch_size, num_workers, pin_memory = self._get_data_params()
        
        return DataLoader(
            dataset=OperatorDataset(cfg=self.cfg),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        batch_size, num_workers, pin_memory = self._get_data_params()
        
        return DataLoader(
            dataset=OperatorDataset(cfg=self.cfg),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[OperatorData]):
        return du.concat_data(batch)

    def teardown(self, stage=None):
        pass
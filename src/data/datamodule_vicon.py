import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig

import src.data.data_utils as du
from src.data.dataloader import CycleDataLooper, DataLooper


class ViconDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def dataloader_vicon(self, cfg):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """

        def collate_fn(data_list: list[dict]):
            vicon_data_list = [item["data"] for item in data_list]
            labels = [item["label"] for item in data_list]

            combined_data = du.concat_data(vicon_data_list)
            combined_labels = torch.cat(labels, dim=0)

            return {"data": combined_data, "label": combined_labels}

        trainset = hydra.utils.instantiate(cfg.dataset)
        return DataLooper(trainset, cfg, batch_size=cfg.batch_size_per_process, collate_fn=collate_fn)

    def train_dataloader(self):
        # cycle through the dataloaders of the different splits
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            # todo in the future: use instantiate to get the dataloader
            dataloopers.append(self.dataloader_vicon(cfg))
        return CycleDataLooper(dataloopers)  # return a single cycle dataloader

    def val_dataloader(self):
        """
        Create and return the validation dataloader.
        :return: a list of dataloopers for validation
        """
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.valid.items()):
            print(f"valid dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            dataloopers.append(self.dataloader_vicon(cfg))
        return dataloopers  # return a list of dataloopers for separate validation

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass

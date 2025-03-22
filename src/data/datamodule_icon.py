from lightning import LightningDataModule
from omegaconf import DictConfig

import src.data.data_utils as du
from src.data.dataloader import CycleDataLooper, DataLooper
from src.data.datasets.dummy_icon import IconData, IconDataset


class IconDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def dataloader_icon(self, cfg):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """

        def collate_fn(data_list: list[IconData]):
            data = du.concat_data(data_list)
            return data

        trainset = IconDataset(cfg=cfg)
        return DataLooper(trainset, cfg, batch_size=cfg.batch_size_per_process, collate_fn=collate_fn)

    def train_dataloader(self):
        # cycle through the dataloaders of the different splits
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            # todo in the future: use instantiate to get the dataloader
            dataloopers.append(self.dataloader_icon(cfg))
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
            dataloopers.append(self.dataloader_icon(cfg))
        return dataloopers  # return a list of dataloopers for separate validation

    def test_dataloader(self):
        pass

    def _collate_fn(self, batch: list[IconData]):
        return du.concat_data(batch)

    def teardown(self, stage=None):
        pass

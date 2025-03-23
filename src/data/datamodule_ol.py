import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig

import src.data.data_utils as du
from src.data.data_utils import OperatorData
from src.data.dataloader import CycleDataLooper, DataLooper


class OperatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def dataloader_ol(self, cfg):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """

        def collate_fn(data_list: list[OperatorData]):
            data = du.concat_data(data_list)
            return data

        # use instantiate to get the dataset, since different config may use different Dataset class
        # you can also explicitly import the Dataset class and define it here to get more flexibility and readability
        trainset = hydra.utils.instantiate(cfg.dataset)

        return DataLooper(trainset, cfg, batch_size=cfg.batch_size_per_gpu, collate_fn=collate_fn)

    def train_dataloader(self):
        # cycle through the dataloaders of the different splits
        dataloopers = []
        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
            # todo in the future: use instantiate to get the dataloader
            dataloopers.append(self.dataloader_ol(cfg))
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
            dataloopers.append(self.dataloader_ol(cfg))
        return dataloopers  # return a list of dataloopers for separate validation

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass

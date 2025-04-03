import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import src.data.data_utils as du
from src.data.dataloader import CycleLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # <- call the data generation script here
        print("Preparing data...")

        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"\t{k}: {v}")
            # instantiate the datasets only on Rank 0, to cache the data in disk
            hydra.utils.instantiate(cfg.dataset)

        for i, (key, cfg) in enumerate(self.cfg.data.valid.items()):
            print(f"valid dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"\t{k}: {v}")
            hydra.utils.instantiate(cfg.dataset)

    def setup(self, stage: str | None = None) -> None:
        """
        called on each process on GPU
        """
        # TODO: move dataset initialization here
        pass

    def dataloader(self, cfg):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """

        def collate_fn(raw_list: list[dict]):
            data_list = [item["data"] for item in raw_list]
            label_list = [item["label"] for item in raw_list]

            combined_data = du.concat_data(data_list)
            combined_labels = du.concat_data(label_list)

            return {"data": combined_data, "label": combined_labels}

        # use instantiate to get the dataset, since different config may use different Dataset class
        dataset = hydra.utils.instantiate(cfg.dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size_per_gpu,
            shuffle=True,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            collate_fn=collate_fn,
            drop_last=True,
        )

    def train_dataloader(self):
        """
        return a single cycle dataloader
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.train.items()):
            dataloaders.append(self.dataloader(cfg))
        return CycleLoader(dataloaders)

    def val_dataloader(self):
        """
        return a list of dataloopers for separate validation
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.valid.items()):
            dataloaders.append(self.dataloader(cfg))
        return dataloaders

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass

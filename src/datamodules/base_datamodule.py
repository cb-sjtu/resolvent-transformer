import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.dataloader_utils import CycleLoader, collate_fn, get_worker_seed_fn


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

    def train_dataloader_from_cfg(self, cfg):
        """
        return a DataLoader for training
        """
        # use instantiate to get the dataset, since different config may use different Dataset class
        dataset = hydra.utils.instantiate(cfg.dataset)

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": collate_fn,
        }

        # if enable_device_seed: seeds vary across different devices
        # it's safe to use global_rank even if not distributed
        # if not enable_device_seed: seeds are shared across different devices (with worker-specific variations)
        # essentially doing nothing
        worker_init_fn = get_worker_seed_fn(
            base_seed=None,
            rank=self.trainer.global_rank,
            enable_device_seed=cfg.enable_device_seed,
            print_info=cfg.name,
            print_lv=self.cfg.print_lv,
        )

        if torch.distributed.is_initialized():
            # if distributed, use DistributedSampler
            # we will wrap the dataloader in CycleLoader,
            # therefore lightning cannot automatically handle DistributedSampler
            return DataLoader(
                dataset=dataset,
                sampler=DistributedSampler(dataset=dataset, shuffle=True, drop_last=True),
                worker_init_fn=worker_init_fn,
                **common_kwargs,
            )
        else:
            # if not distributed, a plain DataLoader is enough
            return DataLoader(
                dataset=dataset, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn, **common_kwargs
            )

    def valid_test_dataloader_from_cfg(self, cfg):
        """
        return a DataLoader for validation or test
        """
        # use instantiate to get the dataset, since different config may use different Dataset class
        dataset = hydra.utils.instantiate(cfg.dataset)

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": collate_fn,
        }

        # always use global_rank
        worker_init_fn = get_worker_seed_fn(
            base_seed=cfg.base_seed,
            rank=self.trainer.global_rank,
            enable_device_seed=cfg.enable_device_seed,
            print_info=cfg.name,
            print_lv=self.cfg.print_lv,
        )

        # Since we're not using CycleLoader here, we can rely on Lightning's built-in handling of DistributedSampler
        return DataLoader(
            dataset=dataset,
            shuffle=False,  # careful: different from train_dataloader
            drop_last=False,  # careful: different from train_dataloader
            worker_init_fn=worker_init_fn,
            **common_kwargs,
        )

    def train_dataloader(self):
        """
        return a single cycle dataloader
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.train.items()):
            dataloaders.append(self.train_dataloader_from_cfg(cfg))
        return CycleLoader(dataloaders)

    def val_dataloader(self):
        """
        return a list of dataloaders for separate validation
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.valid.items()):
            dataloaders.append(self.valid_test_dataloader_from_cfg(cfg))
        return dataloaders  # don't wrap with CycleLoader

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from . import dataloader_utils as dlu


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
        dataset = hydra.utils.instantiate(cfg.dataset)

        # generator will only be created once in the very begining when global_step = 0
        # In our implementation, we can have different random states during the whole training process
        # what happens when we call iter(dataloader)?
        # according to https://discuss.pytorch.org/t/dataloader-persistent-workers-usage/189329
        # if not using persistent workers (by default), new workers will be created when dataloader.__iter__() is called
        # and the generator is used to initialize the worker seeds (which is different from last __iter__() call)
        # setting torch seeds inside worker_init_fn() did not work as expected in our practice.
        # we never tested persistent workers, so we don't know if it works.

        generator = dlu.get_dataloader_rng(
            base_seed=cfg.base_seed,
            enable_device_seed=cfg.enable_device_seed,
            print_info=f"step = {self.trainer.global_step}, train: {cfg.name}",
            print_lv=self.cfg.print_lv,
        )

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": dlu.collate_fn,
            "generator": generator,
        }

        if torch.distributed.is_initialized():
            # if distributed, use DistributedSampler
            # we will wrap the dataloader in CycleLoader,
            # therefore lightning cannot automatically handle DistributedSampler and epoch management
            # use cfg.base_seed as seed. DistributedSampler will add epochs to the seed when __iter__() is called
            sampler = DistributedSampler(dataset=dataset, shuffle=True, seed=cfg.base_seed, drop_last=True)
            dataloader = DataLoader(dataset=dataset, sampler=sampler, **common_kwargs)
            return dataloader, sampler
        else:
            # if not distributed, a plain sampler will suffice
            dataloader = DataLoader(dataset=dataset, shuffle=True, drop_last=True, **common_kwargs)
            return dataloader, None

    def valid_test_dataloader_from_cfg(self, cfg):
        """
        return a DataLoader for validation or test
        """
        dataset = hydra.utils.instantiate(cfg.dataset)

        generator = dlu.get_dataloader_rng(
            base_seed=cfg.base_seed,
            enable_device_seed=cfg.enable_device_seed,
            print_info=f"step = {self.trainer.global_step}, valid: {cfg.name}",
            print_lv=self.cfg.print_lv,
        )

        # generator will only be created once in the very begining when global_step = 0
        # but lightning can somehow use the generator to generate the same random states across validation epochs
        # Don't pass a worker_init_fn into valid_test_dataloader!
        # we found that even pass "lambda worker_id: None" into worker_init_fn
        # will cause different random states across validation epochs.
        # we never tested persistent workers, so we don't know if it works.

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": dlu.collate_fn,
            "generator": generator,
        }

        # We can rely on Lightning's built-in handling of DistributedSampler for validation/test
        return DataLoader(
            dataset=dataset,
            shuffle=False,  # careful: different from train_dataloader
            drop_last=False,  # careful: different from train_dataloader
            **common_kwargs,
        )

    def train_dataloader(self):
        """
        return a single cycle dataloader
        """
        dataloaders = []
        samplers = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.train.items()):
            dataloader, sampler = self.train_dataloader_from_cfg(cfg)
            dataloaders.append(dataloader)
            samplers.append(sampler)
        return dlu.CycleLoader(dataloaders, samplers)

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

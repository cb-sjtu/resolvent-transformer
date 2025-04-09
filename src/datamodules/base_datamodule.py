import random
from functools import partial

import hydra
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

import src.datasets.data_utils as du
from src.datamodules.dataloader import CycleLoader


def custom_worker_seed_fn(worker_id: int, rank: int, base_seed: int | None, dataset_name: str, print_seed: bool):
    """
    custom seed for each worker
    seeds ALWAYS vary across different workers on the same device
    case 0: base_seed is None, rank = 0
            seeds vary across different calls of this function, but are shared across different devices
            essentially doing nothing, i.e. worker_init_fn = None (except for numpy and random seeding and printing)
            this is usually for training, but we want to keep the same seed across different devices
            so that the (random) reshape can be synchronized across different devices (for acceleration)
    case 1: base_seed is None, rank = global_rank
            seeds vary across different calls of this function, and vary across different devices
            this is usually for training with full randomness
    case 3: base_seed is not None, rank = 0
            seeds are fixed across different calls of this function and different devices
            this is not common
    case 4: base_seed is not None, rank = global_rank
            seeds are fixed across different calls of this function, but vary across different devices
            this is usually for validation/testing, so different epochs can be fairly compared
            validation/testing dataset should be deterministic in general, but some cases require RNG within the dataset
            if RNG is used, validation/testing results will not be reproducible if batches or the num_workers is changed
    """
    if print_seed:
        print(
            f"dataset: {dataset_name}, rank: {rank}, worker_id: {worker_id}, "
            f"original initial_seed: {torch.initial_seed()}",
            flush=True,
        )

    # worker_seed should include the base_seed and the worker_id
    worker_seed = torch.initial_seed() if base_seed is None else (base_seed + worker_id)
    seed = (rank * 1000 + worker_seed) % 0xFFFF_FFFF_FFFF_FFFF
    torch.manual_seed(seed)
    np.random.seed(seed % 0xFFFF_FFFF)
    random.seed(seed % 0xFFFF_FFFF)
    if print_seed:
        print(
            f"dataset: {dataset_name}, rank: {rank}, worker_id: {worker_id}, "
            f"updated initial_seed: {torch.initial_seed()}",
            flush=True,
        )


def collate_fn(raw_list: list[dict]):
    data_list = [item["data"] for item in raw_list]
    label_list = [item["label"] for item in raw_list]

    combined_data = du.concat_data(data_list)
    combined_labels = du.concat_data(label_list)

    return {"data": combined_data, "label": combined_labels}


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

        # if random_across_devices: seeds vary across different devices
        # it's safe to use global_rank even if not distributed
        # if not random_across_devices: seeds are shared across different devices (with worker-specific variations)
        # essentially doing nothing
        worker_init_fn = partial(
            custom_worker_seed_fn,
            rank=self.trainer.global_rank if cfg.random_across_devices else 0,
            base_seed=None,
            dataset_name=cfg.name,
            print_seed=self.cfg.print_lv >= 2,
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
        worker_init_fn = partial(
            custom_worker_seed_fn,
            rank=self.trainer.global_rank,
            base_seed=cfg.base_seed,
            dataset_name=cfg.name,
            print_seed=self.cfg.print_lv >= 2,
        )

        # Since we're not using CycleLoader here, we can rely on Lightning's built-in handling of DistributedSampler
        return DataLoader(
            dataset=dataset,
            shuffle=False,
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

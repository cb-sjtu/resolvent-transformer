import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

import src.datasets.data_utils as du
from src.datamodules.dataloader import CycleLoader


def get_worker_init_fn(rank: int, seed: int = None):
    """
    Returns a function that initializes the worker with a random seed.
    This is important to ensure that each worker gets a different seed
    and therefore different data.
    """
    if seed is None:  # if no seed is provided, use a random seed
        seed = torch.randint(0, 0xFFFF_FFFF, (1,)).item()
    seed = seed % 0xFFFF_FFFF  # make sure the seed is in the range of 0 to 0xffff_ffff

    def worker_init_fn(worker_id):
        # set the random seed for each worker
        # max seed: 0xffff_ffff_ffff_ffff
        torch.manual_seed((rank * 0xFFFF + worker_id) * 0xFFFF_FFFF + seed)

    return worker_init_fn


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

        self.need_distributed = "ddp" in cfg.trainer.strategy

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

    def dataloader_from_cfg(self, cfg, skip_distributed: bool = False):
        """
        geometry will be mixed in one batch.
        return: datalooper.
        """
        # use instantiate to get the dataset, since different config may use different Dataset class
        dataset = hydra.utils.instantiate(cfg.dataset)

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": collate_fn,
        }

        if self.need_distributed and not skip_distributed:
            rank = self.trainer.global_rank
            worker_init_fn = get_worker_init_fn(rank) if cfg.random_across_devices else None
            return DataLoader(
                dataset=dataset,
                sampler=DistributedSampler(
                    dataset=dataset,
                    shuffle=True,
                    drop_last=True,
                ),
                worker_init_fn=worker_init_fn,
                **common_kwargs,
            )
        else:
            return DataLoader(
                dataset=dataset,
                shuffle=True,
                drop_last=True,
                **common_kwargs,
            )

    def train_dataloader(self):
        """
        return a single cycle dataloader
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.train.items()):
            dataloaders.append(self.dataloader_from_cfg(cfg))
        return CycleLoader(dataloaders)

    def val_dataloader(self):
        """
        return a list of dataloopers for separate validation
        """
        dataloaders = []
        for _i, (_key, cfg) in enumerate(self.cfg.data.valid.items()):
            dataloaders.append(self.dataloader_from_cfg(cfg, skip_distributed=True))
            #! Note that we skip distributed sampler for validation since `L.Trainer` will take care if it by default
        return dataloaders

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass

import random

import numpy as np
import torch

import src.datasets.data_utils as du


def custom_worker_seed_fn(worker_id: int, rank: int, base_seed: int | None, dataset_name: str, print_seed: bool):
    """
    custom seed for each worker, use as worker_init_fn in DataLoader
    This function is called for each worker every time when it is initialized

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
    # in pytorch, torch.initial_seed() = base_seed + worker_id
    # base_seed varies in different calls, but is shared across different devices by default
    # when Dataloader is created, it gets a RNG derived from the main process (or you can manually pass one into it)
    # this RNG is shared across devices by default (if not manually set with rank)
    # then every time when __iter__() is called, Dataloader use this RNG to generate (new) base_seed for each worker
    original_seed = torch.initial_seed()
    # worker_seed = base_seed + worker_id
    worker_seed = torch.initial_seed() if base_seed is None else (base_seed + worker_id)
    seed = (rank * 1000 + worker_seed) % 0xFFFF_FFFF_FFFF_FFFF
    torch.manual_seed(seed)
    np.random.seed(seed % 0xFFFF_FFFF)
    random.seed(seed % 0xFFFF_FFFF)
    if print_seed:
        print(
            f"dataset: {dataset_name}, rank: {rank}, worker_id: {worker_id}, "
            f"original initial_seed: {original_seed}, "
            f"updated initial_seed: {torch.initial_seed()}",
            flush=True,
        )


def collate_fn(raw_list: list[dict]):
    data_list = [item["data"] for item in raw_list]
    label_list = [item["label"] for item in raw_list]

    combined_data = du.concat_data(data_list)
    combined_labels = du.concat_data(label_list)

    return {"data": combined_data, "label": combined_labels}


# what happens when we call iter(dataloader)?
# according to https://discuss.pytorch.org/t/dataloader-persistent-workers-usage/189329
# if not using persistent workers (by default), new workers will be created when dataloader.__iter__() is called
# and call worker_init_fn() (again) to initialize the worker


class CycleLoader:
    """
    A class that cycles through multiple DataLoader instances in the order of
    step 1: dataloader 1
    step 2: dataloader 2
    step 3: dataloader 3
    step 4: dataloader 1
    step 5: dataloader 2
    step 6: dataloader 3
    ...

    This class takes a list of DataLoader instances and creates an iterator that cycles through
    them sequentially. When one DataLoader is exhausted, it is reset and the cycle continues.

    This CycleLoader should never raise StopIteration. Therefore you can also wrap a single DataLoader
    with this class to create an infinite iterator.

    Attributes:
        dataloaders (list): A list of DataLoader instances to cycle through

    Methods:
        __init__(dataloaders): Initializes the CycleLoader with a list of DataLoaders
        __iter__(): Initializes iterators for each DataLoader and returns self
        __next__(): Returns the next batch from the current DataLoader, cycling through them
                    indefinitely. Resets exhausted DataLoaders automatically.
    """

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        # Keep an active iterator per sub-loader
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.idx = 0  # which loader we're pulling from
        return self

    def __next__(self):
        try:
            # Attempt to get a batch from the current loader
            batch = next(self.iterators[self.idx])
            # Move to the next loader
            self.idx = (self.idx + 1) % len(self.dataloaders)
            return batch
        except StopIteration:
            # Current loader is exhausted; reset its iterator
            self.iterators[self.idx] = iter(self.dataloaders[self.idx])
            # Try again from the newly-reset iterator at the same index
            batch = next(self.iterators[self.idx])
            # Here we didn't use recursive call to avoid infinite loop
            # If StopIteration is raised again, it means the dataloader is not enough for one batch
            # In this case, we will raise StopIteration
            self.idx = (self.idx + 1) % len(self.dataloaders)
            return batch

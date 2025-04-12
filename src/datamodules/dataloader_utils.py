import random
from collections.abc import Callable

import numpy as np
import torch

import src.datasets.data_utils as du


def get_worker_seed_fn(
    base_seed: int | None, rank: int, enable_device_seed: bool, print_info: str, print_lv: int = 0
) -> Callable:
    """
    Custom seed function for each worker (used as `worker_init_fn` in a DataLoader).

    This function is called every time a worker is initialized. **Seeds always vary**
    across different workers on the same device.

    Args:
        worker_id (int): ID of the worker process.
        rank (int): Global rank (0-based index) of the process.
        base_seed (int | None): Base seed to initialize the random generator.
            If `None`, a dynamic seed is used.
        print_info (str, optional): Optional string to print for debugging or
            informational purposes.
        print_lv (int, optional): Optional integer to control the verbosity of
            the print statement. Print only when `print_lv >= 2`.

    The function behaves differently depending on the combination of `base_seed`
    and `rank`, identified by the following cases:

    **case 0**:
        - base_seed is `None`, `enable_device_seed = False`
        - seeds vary across different calls of this function, but are shared across different devices
        essentially doing nothing, i.e. `worker_init_fn = None` (except for numpy and random seeding and printing)
        this is usually for training, but we want to keep the same seed across different devices
        so that the (random) reshape can be synchronized across different devices (for acceleration)

    **case 1**:
        - base_seed is `None`, `enable_device_seed = True`
        - seeds vary across different calls of this function, and vary across different devices
        this is usually for training with full randomness

    **case 2**:
        - base_seed is not `None`, `enable_device_seed = False`
        - seeds are fixed across different calls of this function and different devices
        this is not common

    **case 3**:
        - base_seed is not `None`, `enable_device_seed = True`
        - seeds are fixed across different calls of this function, but vary across different devices
        this is usually for validation/testing, so different epochs can be fairly compared
        validation/testing dataset should be deterministic in general, but some cases require RNG within the dataset
        if RNG is used, validation/testing results will not be reproducible if batches or the num_workers is changed
    """
    # in pytorch, torch.initial_seed() = base_seed + worker_id
    # see https://github.com/pytorch/pytorch/blob/a6933a1c423261de4e0c47387b6b83869f869aa1/torch/utils/data/dataloader.py#L1147
    # base_seed varies in different calls, but is shared across different devices by default
    # when Dataloader is created, it gets a RNG derived from the main process (or you can manually pass one into it)
    # this RNG is shared across devices by default (if not manually set with rank)
    # then every time when __iter__() is called, Dataloader use this RNG to generate (new) base_seed for each worker

    def worker_init_fn(worker_id: int) -> None:
        original_seed = torch.initial_seed()  # = base_seed + worker_id
        # worker_seed = base_seed + worker_id
        seed_suffix = (original_seed - worker_id) % 0x1_0000_0000 if base_seed is None else base_seed % 0x1_0000_0000
        seed = (
            seed_suffix * 0x1_0000_0000  # ！
            + (rank if enable_device_seed else 0xFFFF) * 0x1_0000  # ！
            + worker_id
        )  # {original_seed_suffix}_{rank}_{worker_id}
        torch.manual_seed(seed)
        random.seed(seed)

        np_seed = torch.randint(0, 0xFFFF_FFFF, (1,), dtype=torch.int64).item()
        # alternatively, we may use `seed` and `worker_id` to re-generate a random seed
        # https://github.com/pytorch/pytorch/blob/a6933a1c423261de4e0c47387b6b83869f869aa1/torch/utils/data/_utils/worker.py#L176
        np.random.seed(np_seed)

        if print_lv >= 2:
            print(
                f"r/w=[0x{rank:04x}/0x{worker_id:04x}]\t"
                f"original: 0x{original_seed:016x}\t"
                f"updated: 0x{seed:016x} ({print_info})",
                flush=True,
            )

    return worker_init_fn


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

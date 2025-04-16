import torch
from torch.utils.data import DataLoader, DistributedSampler

import src.datasets.data_utils as du

# We follow the practice in
# https://pytorch.org/docs/stable/notes/randomness.html#dataloader
# to set the generator for the dataloader


def get_dataloader_rng(
    base_seed: int,
    enable_device_seed: bool,
    print_info: str,
    print_lv: int,
) -> torch.Generator:
    """
    Get the RNG for the dataloader.

    Args:
        base_seed (int): The mandatory base seed for the dataloader.
                         If you want a dynamic seed (which is not recommended), set it out of this function.
        enable_device_seed (bool): whether to use per-device seed.
                        If True, augment the seed with the device rank.
        print_info (str): the info to print.
        print_lv (int): the verbosity level.

    Returns:
        torch.Generator: The RNG for the dataloader.
    """
    generator = torch.Generator()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    seed = base_seed + rank if enable_device_seed else base_seed
    generator.manual_seed(seed)
    if print_lv >= 2:
        print(f"dataloader rng, rank=[0x{rank:04x}]\tseed=[0x{seed:016x}]\t({print_info})", flush=True)
    return generator


def collate_fn(raw_list: list[dict]):
    data_list = [item["data"] for item in raw_list]
    label_list = [item["label"] for item in raw_list]

    combined_data = du.concat_data(data_list)
    combined_labels = du.concat_data(label_list)

    return {"data": combined_data, "label": combined_labels}


class CycleLoader:
    """
    This class takes a list of dataloader instances and creates an iterator that cycles through
    them sequentially:
    step 1: dataloader 1
    step 2: dataloader 2
    step 3: dataloader 3
    step 4: dataloader 1
    step 5: dataloader 2
    step 6: dataloader 3
    ...
    When one dataloader is exhausted, it is reset and the cycle continues.

    This CycleLoader should never raise StopIteration. Therefore you can also wrap a single DataLoader
    with this class to create an infinite iterator.

    Attributes:
        dataloaders (list): A list of dataLoader instances to cycle through.
        samplers (list): A list of Sampler instances for each dataLoader.
        see __init__ for more details.

    Methods:
        __init__(dataloaders, samplers): Initializes the CycleLoader with a list of dataloaders and samplers
        __iter__(): Initializes iterators for each dataloader and returns self
        __next__(): Returns the next batch from the current dataloader, cycling through them
                    indefinitely. Resets exhausted dataloaders automatically.
    """

    def __init__(
        self,
        dataloaders: list[DataLoader | "CycleLoader"],
        samplers: list[DistributedSampler | None],
    ):
        """
        The elements in the `dataloaders` list can be a mix of DataLoaders and CycleLoaders.

        If an element is a DataLoader using DistributedSampler,
        we must explicitly provide the DistributedSampler to the __init__ function to manually set the epoch.
        This is necessary since DistributedSampler requires manual epoch management. See example in
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        Lightning's automatic epoch handling is bypassed when dataloaders are wrapped in CycleLoader.

        Set sampler = None for other elements, including:
        - DataLoaders with other types of samplers, as they do not need manual epoch management.
        - CycleLoaders, as they should be able to handle the epoch management by themselves.
        """
        self.dataloaders = dataloaders
        self.samplers = samplers
        self.epochs = [0] * len(dataloaders)

    def __iter__(self):
        # Keep an active iterator per sub-loader

        # at the beginning of each epoch before creating the DataLoader iterator

        for i, sampler in enumerate(self.samplers):
            if sampler is not None:
                sampler.set_epoch(self.epochs[i])
            self.epochs[i] += 1

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

            # at the beginning of each epoch before creating the DataLoader iterator
            if self.samplers[self.idx] is not None:
                self.samplers[self.idx].set_epoch(self.epochs[self.idx])
            self.epochs[self.idx] += 1
            self.iterators[self.idx] = iter(self.dataloaders[self.idx])

            # Try again from the newly-reset iterator at the same index
            batch = next(self.iterators[self.idx])
            # Here we didn't use recursive call to avoid infinite loop
            # If StopIteration is raised again, it means the dataloader is not enough for one batch
            # In this case, we will raise StopIteration
            self.idx = (self.idx + 1) % len(self.dataloaders)
            return batch

from omegaconf import DictConfig
from torch.utils.data import DataLoader


class DataLooper:
    def __init__(self, dataset, cfg: DictConfig, batch_size: int, collate_fn=None):
        """
        cfg: num_workers, infinite, print_info, pin_memory
        explicitly pass batch_size, since the batch_size passed into model can be different from the one in dataloader
        """
        self.dataset = dataset
        self.cfg = cfg
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.data_loader = self.get_dataloader()
        self.data_iter = iter(self.data_loader)
        self.data_iter_num = 0

    def get_dataloader(self):
        drop_last = self.batch_size is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Ensure reshuffling occurs when creating each new dataloader
            drop_last=drop_last,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=self.collate_fn,
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = next(self.data_iter)
        except StopIteration:
            if not self.cfg.infinite:
                raise StopIteration  # Stop the iteration  # noqa: B904
            if self.cfg.print_info:
                print(f"{self.dataset.name} reached end of data loader, restart {self.data_iter_num}")
            self.data_loader = self.get_dataloader()
            self.data_iter = iter(self.data_loader)
            self.data_iter_num += 1
            out = next(self.data_iter)
        return out


class CycleDataLooper:
    def __init__(self, dataloopers):
        """
        Initialize with a list of DataLooper instances.
        """
        self.dataloopers = dataloopers
        self.current_looper_idx = 0  # Track the current DataLooper to use
        self.looper_count = len(self.dataloopers)

    def __iter__(self):
        """
        Make this class an iterator by returning self.
        """
        return self

    def __next__(self):
        """
        Cycle through the dataloaders of the DataLooper instances.
        """
        if self.looper_count == 0:
            raise StopIteration("No DataLooper instances provided.")

        # Get the current DataLooper
        current_looper = self.dataloopers[self.current_looper_idx]

        try:
            # Try to get the next batch from the current DataLooper
            out = next(current_looper)
        except StopIteration:
            # If the current DataLooper is exhausted, move to the next one
            self.current_looper_idx = (self.current_looper_idx + 1) % self.looper_count
            current_looper = self.dataloopers[self.current_looper_idx]
            out = next(current_looper)

        # Move to the next DataLooper for the next call
        self.current_looper_idx = (self.current_looper_idx + 1) % self.looper_count
        return out

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

#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################


import transformers as tfs
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.datamodules.base_datamodule import BaseDataModule


class ProcessDatasetWrapper(Dataset):
    """
    Wrapper dataset that applies post-processing to individual samples.
    This allows the processing to run in parallel with DataLoader workers.
    """

    def __init__(self, dataset, image_processor):
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process images in the worker process
        item["processed_images"] = self.process_image(item["raw_images"])

        return item

    def process_image(self, image):
        """
        Process image with image_processor
        image: [..., c, h, w], int between 0 and 255
        return [..., c, h', w']
        """
        reshaped_image = image.reshape(-1, *image.shape[-3:])  # (..., c, h, w) -> (n, c, h, w)
        processed_image = self.image_processor(reshaped_image, return_tensors="pt")["pixel_values"]
        # (n, c, h', w') -> (..., c, h', w')
        processed_image = processed_image.reshape(image.shape[:-3] + processed_image.shape[-3:])
        return processed_image


class WrapperDataModule(BaseDataModule):
    """
    ImageProcessor always returns tensors on CPUs, even if the input is on GPUs.
    Therefore, we put the image processing in the DataModule instead of the models.
    This is a simple example of how to post-process individual samples in the DataModule.
    You can also implement more complex processing in the wrapper.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # for example, using HuggingFace ImageProcessor to process images
        vision_tower_name = self.cfg.data.image_encoder.vision_tower_name
        self.image_processor = tfs.AutoImageProcessor.from_pretrained(vision_tower_name)

    def get_train_dataset_from_cfg(self, cfg):
        """
        Override to wrap dataset with processing wrapper
        """
        base_result = super().get_train_dataset_from_cfg(cfg)
        dataset = base_result["dataset"]

        # Wrap with processing dataset
        processed_dataset = ProcessDatasetWrapper(dataset, self.image_processor)

        return {"dataset": processed_dataset, "cfg": cfg}

    def get_valid_test_dataset_from_cfg(self, cfg):
        """
        Override to wrap dataset with processing wrapper
        """
        base_result = super().get_valid_test_dataset_from_cfg(cfg)
        dataset = base_result["dataset"]

        # Wrap with processing dataset
        processed_dataset = ProcessDatasetWrapper(dataset, self.image_processor)

        return {"dataset": processed_dataset, "cfg": cfg}

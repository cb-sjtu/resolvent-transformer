"""
this file contains custom utils not included in the original template but useful for our research.
For project-specific utils, please create new files like xxx_project_utils.py
"""

from omegaconf import DictConfig


def get_dataset_name(data_cfg: DictConfig, dataloader_idx: int) -> str:
    """
    usually cfg.data.valid/test is a dict of datasets
    this function returns the name of the dataset for the given dataloader index
    """
    key = list(data_cfg.keys())[dataloader_idx]
    dataset_name = data_cfg[key].name
    return dataset_name


def get_batch_description(batch: dict) -> list[str]:
    """
    get the description of the batch
    for example:
    batch = {
        "data": Data(description=["D1", "D2"]),
        "label": Data(description=["L1", "L2"]),
    }
    will return:
    ["data: D1, label: L1", "data: D2, label: L2"]
    """
    description = []
    for k, v in batch.items():
        # this should always be true, but just in case
        if hasattr(v, "description") and isinstance(v.description, list):
            description.append([f"{k}: {d}" for d in v.description])

    assert len(description) > 0, "no description found in the batch"

    concat_description = []
    for i in range(len(description[0])):
        concat_description.append(", ".join([d[i] for d in description]))

    return concat_description

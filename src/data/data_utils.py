import os
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def dict_to_namedtuple(d: dict, name="Data"):
    """Convert a dictionary to a namedtuple recursively."""
    # Recursively apply to dictionaries within the dictionary
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namedtuple(value, name=key.capitalize())
    # Create the namedtuple type and instantiate it
    TupleType = namedtuple(name, d.keys())
    return TupleType(**d)


def slice_nodes_set(nodal_value: torch.Tensor, set_idx: torch.Tensor) -> torch.Tensor:
    """
    nodal_value: (bs, nnode, ...)
    set_idx: (bs, n_set)
    return: (bs, n_set, ...)
    """
    if isinstance(nodal_value, torch.Tensor):
        batch_idx = torch.arange(nodal_value.shape[0]).unsqueeze(1)  # shape: (bs, 1)
        nodal_value_set = nodal_value[batch_idx, set_idx, ...]  # (bs, n_set, ...)
        return nodal_value_set
    elif isinstance(nodal_value, np.ndarray):
        batch_idx = np.arange(nodal_value.shape[0]).reshape(-1, 1)  # shape: (bs, 1)
        nodal_value_set = nodal_value[batch_idx, set_idx, ...]  # (bs, n_set, ...)
        return nodal_value_set
    else:
        raise ValueError(f"Unknown type: {type(nodal_value)}")


class DataBase:
    def to(self, device):
        # Iterates over all attributes of the instance, moving each to the specified device
        # create a new object so the original data is not changed
        new_data = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(new_data, attr, value.to(device))
            else:
                setattr(new_data, attr, value)
        return new_data

    def to_numpy(self):
        new_data = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(new_data, attr, value.detach().cpu().numpy())
            else:
                setattr(new_data, attr, value)
        return new_data

    def to_tensor(self):
        new_data = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(new_data, attr, torch.tensor(value))
            else:
                setattr(new_data, attr, value)
        return new_data

    def get_one_batch(self, bid, keep_dim=False):
        """
        return new data with bid-th batch, can keep batch dim or not
        """
        new_data = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                new_value = value[bid : bid + 1] if keep_dim else value[bid]
                setattr(new_data, attr, new_value)
            elif isinstance(value, list | tuple):
                new_value = [value[bid]] if keep_dim else value[bid]
                setattr(new_data, attr, new_value)
            else:
                setattr(new_data, attr, value)
        return new_data

    def get_slice_batch(self, bid_list):
        """
        return new data with bid-th batch, can keep batch dim or not
        """
        new_data = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                setattr(new_data, attr, value[bid_list])
            elif isinstance(value, list | tuple):
                setattr(new_data, attr, [value[i] for i in bid_list])
            else:
                setattr(new_data, attr, value)
        return new_data

    def get_shape(self):
        data_shape = type(self)()
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                setattr(data_shape, attr, value.shape)
            elif isinstance(value, list | tuple | str):
                setattr(data_shape, attr, len(value))
            else:
                setattr(data_shape, attr, None)
        return data_shape

    def get_shape_namedtuple(self, exclude_batch=True):
        # return a named tuple with shape of each useful attribute
        # used for identifying whether we need to build a new mask rather than using the existing ones
        data_shape = {}
        for attr, value in self.__dict__.items():
            if attr not in ["problem", "param"] and value is not None:
                if exclude_batch:
                    data_shape[attr] = tuple(value.shape[1:])
                else:
                    data_shape[attr] = tuple(value.shape)
        return dict_to_namedtuple(data_shape)

    def get_print_info(self, print_lv: int = 1) -> str:
        doc = "=" * 20 + "\n"

        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                doc += f"{attr}: type={type(value)}\tshape={value.shape}\tdtype={value.dtype}\tdevice={value.device}\n"
            elif isinstance(value, np.ndarray):
                doc += f"{attr}: type={type(value)}\tshape={value.shape}\tdtype={value.dtype}\n"
            elif isinstance(value, tuple | list):
                doc += f"{attr}: length={len(value)}, {type(value).__name__} of {type(value[0]).__name__}\n"
                if print_lv == 1:
                    if len(value) <= 4:
                        for i in range(len(value)):
                            doc += str(value[i]) + "\n"
                    else:
                        for i in range(2):
                            doc += str(value[i]) + "\n"
                        doc += "...\n"
                        doc += str(value[-2]) + "\n"
                        doc += str(value[-1]) + "\n"
                if print_lv == 2:
                    for i in range(len(value)):
                        doc += str(value[i]) + "\n"
            else:
                doc += f"{attr}: type={type(value)}\t value={str(value)}\n"
        doc += "=" * 20 + "\n"
        return doc


@dataclass
class DataMesh(DataBase):
    # N_t: number of stretches
    # N_e: number of elements (triangles)
    # N_n: number of nodes
    # N_bc: number of boundary nodes
    # N_free: number of free nodes
    # N_ifx1: number of ifx1 nodes
    # N_ifx2: number of ifx2 nodes

    rawpath: str = None
    problem: list[str] = None  # type of material, tuple of strings
    param: list = None  # unknown, keep it None for now
    mesh: np.ndarray = None  # (N_t, N_e, r), (1231, 3)
    X: np.ndarray = None  # (N_t, N_n, 2), (681, 2)
    U: np.ndarray = None  # (N_t, N_n, 2), (11, 681, 2)
    BCset: np.ndarray = None  # (N_t, N_bc), [19, 20, 21, 22, 59, 60, ..., 128, 129, 130]
    FREEset: np.ndarray = None  # (N_t, N_free), (629,)
    IFX1set: np.ndarray = None  # (N_t, N_ifx1), [19, 21, 59, 60, 61, ..., 79, 80, 81, 82]
    IFX2set: np.ndarray = None  # (N_t, N_ifx2), [20, 22, 107, 108, 109, ..., 128, 129, 130]
    IFX: np.ndarray = None  # (N_t,), (11,)
    Xip: np.ndarray = None  # (N_t, N_e, 1, 2), (1231, 1, 2)
    B: np.ndarray = None  # (N_t, N_e, 1, 3, 2), (1231, 1, 3, 2)
    Bcoeff: np.ndarray = None  # (N_t, N_e, 1), (1231, 1)
    F: np.ndarray = None  # (N_t, N_e, 1, 2, 2), (11, 1231, 1, 2, 2)
    P: np.ndarray = None  # (N_t, N_e, 1, 2, 2)
    S: np.ndarray = None  # (N_t, N_e, 1, 2, 2)

    E: np.ndarray = None  # (N_t, N_e, 1, 2, 2)
    C: np.ndarray = None  # (N_t, N_e, 1, 2, 2)
    I: np.ndarray = None  # (N_t, N_e, 1, 2)  # noqa: E741
    grad_I_grad_C: np.ndarray = None  # (N_t, N_e, 1, 2, 2, 2)

    A: np.ndarray = None  # (N_t, N_n, 8, 2, 2) [at most 8 neighbors]
    XI: np.ndarray = None  # (N_t, N_n, 8, 2) [at most 8 neighbors]
    mask: np.ndarray = None  # (N_t, N_n, 8) [mask for A and XI]

    def save(self, filepath: Path) -> None:
        """
        Save all dataclass fields into a dictionary and then pickle it to 'filepath'.
        """

        os.makedirs(filepath.parent, exist_ok=True)
        # Build a dictionary from all dataclass fields
        data_dict = {field: getattr(self, field) for field in self.__dataclass_fields__}

        # Optionally convert rawpath to string if you want consistent typing
        # data_dict["rawpath"] = str(self.rawpath) if self.rawpath is not None else None

        np.save(filepath, data_dict, allow_pickle=True)

    @classmethod
    def load(cls, filepath: Path, rawpath: Path) -> "DataMesh":
        try:
            return cls.load_from_file(filepath)
        except FileNotFoundError:
            data = cls.load_from_raw(rawpath)
            data.save(filepath)
            return data

    @classmethod
    def load_from_file(cls, filepath: Path) -> "DataMesh":
        data_dict = np.load(filepath, allow_pickle=True).item()
        return cls(**data_dict)

    @classmethod
    def load_from_raw(cls, rawpath: Path) -> "DataMesh":
        pass


@dataclass
class DataEqn(DataBase):
    def __init__(self, description=None, A=None, XI=None, mask=None):
        super().__init__()
        self.description = description
        self.A = A
        self.XI = XI
        self.mask = mask


@dataclass
class DataMeshList:
    description: str
    meshes: list[DataMesh]

    def __init__(self, description: str, meshes: list[DataMesh]):
        self.description = description
        self.meshes = meshes

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        return self.meshes[idx]

    def __iter__(self):
        return iter(self.meshes)

    def __next__(self):
        return next(self.meshes)

    def append(self, mesh: DataMesh):
        self.meshes.append(mesh)

    def get_print_info(self, print_lv: int = 1) -> str:
        doc = "*" * 50 + "\n"
        doc += self.description + "\n"
        doc += f"DataMeshList: {len(self.meshes)} meshes\n"
        for i, data in enumerate(self.meshes):
            doc += f"DataMesh {i}:\n"
            doc += data.get_print_info(print_lv)
        doc += "*" * 50
        return doc


def concat_data(datas: Sequence[T], to_tensor=True) -> T:
    """
    concat a sequence of data into a single data
    """
    data = type(datas[0])()

    for attr, value in datas[0].__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(data, attr, torch.concat([getattr(d, attr) for d in datas]))
        elif isinstance(value, np.ndarray):
            setattr(data, attr, np.concatenate([getattr(d, attr) for d in datas]))
        elif isinstance(value, list):
            setattr(data, attr, sum([getattr(d, attr) for d in datas], []))
        elif isinstance(value, tuple):
            setattr(data, attr, tuple(sum([getattr(d, attr) for d in datas], [])))
        else:  # just put them in a list
            setattr(data, attr, [getattr(d, attr) for d in datas])
    data = data.to_tensor() if to_tensor else data
    return data

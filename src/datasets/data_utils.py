from collections import namedtuple
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch

T = TypeVar("T", bound="BaseData")


def dict_to_namedtuple(d: dict[str, Any], name: str = "Data"):
    """Convert a dictionary to a namedtuple recursively."""
    # Recursively apply to dictionaries within the dictionary
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namedtuple(value, name=key.capitalize())
    # Create the namedtuple type and instantiate it
    TupleType = namedtuple(name, d.keys())  # type: ignore
    # https://github.com/python/mypy/issues/9046#issuecomment-649736524
    return TupleType(**d)


class BaseData:
    description: list | tuple | None

    def _apply(self: T, func: Callable[[str, Any], Any], *args, **kwargs) -> T:
        """
        Helper method to apply a function to each attribute of the instance.
        Creates a new instance of the same type and assigns the transformed values.
        """
        new_obj = type(self)()  # This works if all fields have defaults.
        for attr, value in self.__dict__.items():
            setattr(new_obj, attr, func(attr, value, *args, **kwargs))
        return new_obj

    def to(self: T, device: torch.device, *args, **kwargs) -> T:
        """
        Return a new instance with each torch.Tensor moved to the specified device.
        """
        return self._apply(lambda attr, v: v.to(device, *args, **kwargs) if isinstance(v, torch.Tensor) else v)

    def to_numpy(self: T) -> T:
        """
        Return a new instance with torch.Tensors converted to numpy arrays.
        """
        return self._apply(lambda attr, v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)

    def to_tensor(self: T, *args, **kwargs) -> T:
        """
        Return a new instance with numpy arrays converted to torch.Tensors.
        """
        return self._apply(lambda attr, v: torch.tensor(v, *args, **kwargs) if isinstance(v, np.ndarray) else v)

    def get_one_batch(self: T, bid: int, keep_dim=False) -> T:
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

    def get_slice_batch(self: T, bid_list: Sequence[int]) -> T:
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

    def _get_print_info_seq(self, attr: str, value: list | tuple, print_lv: int = 1) -> str:
        """
        print the sequence of data, end with a newline
        """
        doc = ""
        doc += f"{attr}: length={len(value)}, {type(value).__name__} of {type(value[0]).__name__}\n"
        if print_lv == 0 or print_lv == 1:  # print the first 2 and the last 2
            if len(value) <= 4:
                for i in range(len(value)):
                    doc += str(value[i]) + "\n"
            else:
                for i in range(2):
                    doc += str(value[i]) + "\n"
                doc += "...\n"
                doc += str(value[-2]) + "\n"
                doc += str(value[-1]) + "\n"
        if print_lv == 2:  # print all
            for i in range(len(value)):
                doc += str(value[i]) + "\n"
        return doc

    def _get_print_info_lv0(self) -> str:
        """
        print data level 0, end with a newline
        """
        doc = ""
        if not hasattr(self, "description") or self.description is None:
            doc += "no description\n"
        elif isinstance(self.description, list | tuple):
            doc += self._get_print_info_seq("description", self.description, print_lv=0)
        else:
            raise ValueError(f"Unknown type: {type(self.description)}")
        return doc

    def _get_print_info_lv1(self) -> str:
        """
        print data level 1, end with a newline
        """
        doc = ""

        # First print all sequences with print_lv=1
        for attr, value in self.__dict__.items():
            if isinstance(value, tuple | list):
                doc += self._get_print_info_seq(attr, value, print_lv=1)

        # Then print all strings (if any)
        for attr, value in self.__dict__.items():
            if isinstance(value, str):
                doc += f"{attr}: {type(value)} {str(value)}\n"

        # Finally print all tensors/arrays in one line
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor | np.ndarray):
                doc += f"{attr}: {value.shape} {value.dtype} | "
        doc += "\n"
        return doc

    def _get_print_info_lv2(self) -> str:
        """
        print data level 2, end with a newline
        """
        doc = ""
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                doc += f"{attr}: {value.shape} {value.dtype} {value.device}\n"
            elif isinstance(value, np.ndarray):
                doc += f"{attr}: {value.shape} {value.dtype}\n"
            elif isinstance(value, tuple | list):
                doc += self._get_print_info_seq(attr, value, print_lv=2)
            else:
                doc += f"{attr}: type={type(value)} value={str(value)}\n"
        return doc

    def get_print_info(self, print_lv: int = 1, info: str | None = None) -> str:
        """
        print the information of the data, no newline in the end
        print_lv = 0: only print the first 2 and the last 2 of description list, ignore other attributes
        print_lv = 1: print the first 2 and the last 2 of sequence attributes, and all other attributes
        print_lv = 2: print all
        """
        doc = "-" * 10 + f"begin {info + ', ' if info is not None else ''}{self.__class__.__name__} " + "-" * 10 + "\n"

        if print_lv == 0:
            doc += self._get_print_info_lv0()
        elif print_lv == 1:
            doc += self._get_print_info_lv1()
        elif print_lv == 2:
            doc += self._get_print_info_lv2()
        else:
            raise ValueError(f"Unknown print_lv: {print_lv}")

        doc += "-" * 10 + f"end {info + ', ' if info is not None else ''}{self.__class__.__name__} " + "-" * 10
        return doc


def concat_data(datas: Sequence[BaseData], to_tensor=True) -> BaseData:
    """
    concat a sequence of data into a single data
    """
    data = type(datas[0])()  # same type as the elements, could be a child of BaseData

    for attr, value in datas[0].__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(data, attr, torch.concat([getattr(d, attr) for d in datas]))
        elif isinstance(value, np.ndarray):
            setattr(data, attr, np.concatenate([getattr(d, attr) for d in datas]))
        elif isinstance(value, list):
            setattr(data, attr, sum([getattr(d, attr) for d in datas], []))
        elif isinstance(value, tuple):
            setattr(data, attr, tuple(sum([getattr(d, attr) for d in datas], [])))
        elif isinstance(value, str):
            setattr(data, attr, "".join([getattr(d, attr) for d in datas]))
        else:  # just put them in a list
            setattr(data, attr, [getattr(d, attr) for d in datas])
    data = data.to_tensor() if to_tensor else data
    return data


# -------Input Data-------
@dataclass
class OperatorData(BaseData):
    description: list[str] | None = None
    f_samples: torch.Tensor | None = None  # (batch, f_seq_len, f_inout_dim)
    g_inputs: torch.Tensor | None = None  # (batch, g_seq_len, g_in_dim)


@dataclass
class ViconData(BaseData):
    description: list[str] | None = None
    demo_cond: torch.Tensor | None = None
    quest_cond: torch.Tensor | None = None
    demo_qoi: torch.Tensor | None = None


# -------Label Data-------
# split from Input Data to avoid accidental use of label
# also wrap into BaseData for flexible operations
@dataclass
class BaseLabelData(BaseData):
    description: list[str] | None = None
    label: torch.Tensor | None = None

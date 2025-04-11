from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


def dict_to_namedtuple(d: dict, name="Data"):
    """Convert a dictionary to a namedtuple recursively."""
    # Recursively apply to dictionaries within the dictionary
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namedtuple(value, name=key.capitalize())
    # Create the namedtuple type and instantiate it
    TupleType = namedtuple(name, d.keys())
    return TupleType(**d)


class BaseData:
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

    def _get_print_info_seq(self, attr: str, value: list | tuple, print_lv: int = 1) -> str:
        '''
        print the sequence of data
        '''
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

    def get_print_info(self, print_lv: int = 1) -> str:
        '''
        print the information of the data
        print_lv = 0: only print the first 2 and the last 2 of description list, ignore other attributes
        print_lv = 1: print the first 2 and the last 2 of sequence attributes, and all other attributes
        print_lv = 2: print all
        '''
        if print_lv == 0:
            if not hasattr(self, "description") or self.description is None:
                doc = ""
            elif isinstance(self.description, list | tuple):
                doc = self.get_print_info_seq("description", self.description, print_lv=print_lv)
            else:
                raise ValueError(f"Unknown type: {type(self.description)}")
            return doc

        doc = "-" * 20 + "\n"
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                doc += f"{attr}: type={type(value)}\tshape={value.shape}\tdtype={value.dtype}\tdevice={value.device}\n"
            elif isinstance(value, np.ndarray):
                doc += f"{attr}: type={type(value)}\tshape={value.shape}\tdtype={value.dtype}\n"
            elif isinstance(value, tuple | list):
                doc += self._get_print_info_seq(attr, value, print_lv=print_lv)
            else:
                doc += f"{attr}: type={type(value)}\t value={str(value)}\n"
        doc += "-" * 20 + "\n"
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
    description: list[str] = None
    f_samples: torch.Tensor = None  # (batch, f_seq_len, f_inout_dim)
    g_inputs: torch.Tensor = None  # (batch, g_seq_len, g_in_dim)


@dataclass
class ViconData(BaseData):
    description: list[str] = None
    demo_cond: torch.Tensor = None
    quest_cond: torch.Tensor = None
    demo_qoi: torch.Tensor = None


# -------Label Data-------
# split from Input Data to avoid accidental use of label
# also wrap into BaseData for flexible operations
@dataclass
class BaseLabelData(BaseData):
    description: list[str] = None
    label: torch.Tensor = None

#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import h5py
import numpy as np
import torch


class KSDataset:
    def __init__(
        self,
        path: str,
        split: str,
        nt: int,
        nx: int,
        n_input_times: int,
        n_output_times: int,
        min_time_step: int,
        max_time_step: int,
    ):
        """
        path: path to dataset
        split: [train, valid, test]
        nt: temporal resolution
        nx: spatial resolution
        """
        super().__init__()
        self.split = split
        self.dataset = f"pde_{nt}-{nx}"
        self.file = h5py.File(path, "r")
        self.n_traj = self.file[self.split][self.dataset].shape[0]
        self.n_input_times = n_input_times
        self.n_output_times = n_output_times
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        self._build_metadata()

    def _build_metadata(self):
        steps = self.file[self.split][self.dataset].shape[1]
        steps = min(steps, self.max_time_step + 1) - max(0, self.min_time_step)
        windows_per_traj = max(0, steps - self.n_input_times - self.n_output_times + 1)

        self.n_steps_per_traj = steps
        self.n_windows_per_traj = windows_per_traj
        self.time_index_offset = max(self.min_time_step, 0)
        self.length = self.n_traj * self.n_windows_per_traj

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        sample_idx = idx // self.n_windows_per_traj
        time_idx = idx % self.n_windows_per_traj + self.time_index_offset
        time_in_offset = time_idx + self.n_input_times
        input_fields = torch.Tensor(self.file[self.split][self.dataset][sample_idx, time_idx:time_in_offset]).unsqueeze(
            0
        )
        output_fields = torch.Tensor(
            self.file[self.split][self.dataset][sample_idx, time_in_offset : (time_in_offset + self.n_output_times)]
        ).unsqueeze(0)
        x = torch.Tensor(self.file[self.split]["x"][sample_idx])
        input_t = torch.Tensor(self.file[self.split]["t"][sample_idx, time_idx:time_in_offset])
        output_t = torch.Tensor(
            self.file[self.split]["t"][sample_idx, time_in_offset : (time_in_offset + self.n_output_times)]
        )
        description = f"KS equation, input_t={input_t}, output_t={output_t[0:2]}..., x_shape={x.shape}"

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": input_fields,
            "label": output_fields,
        }

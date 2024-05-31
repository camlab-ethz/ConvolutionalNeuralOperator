import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import netCDF4 as nc
from abc import ABC
from typing import Optional

#---------------------------------------------------
# All the datasets (21 of them) are available at: _
#---------------------------------------------------

class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        resolution: Optional[int] = None,
        in_dist: Optional[bool] = True,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        time_input: Optional[bool] = True,
        masked_input: Optional[list] = None
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            in_dist: Whether to use in distribution or out of distribution data.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            time_input: Time in the input channels?
        """
        assert which in ["train", "val", "test"]
        assert resolution is not None and resolution > 0
        assert num_trajectories is not None and num_trajectories > 0
        
        #xprint(resolution, "RES")
        self.resolution = resolution
        self.in_dist = in_dist
        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.time_input = time_input
        
        self.masked_input = masked_input
        if self.masked_input is not None:
            self.mask = torch.tensor(self.masked_input, dtype=torch.float32)
        
        if self.time_input:
            self.in_dim = 3
        else:
            self.in_dim = 2
        self.out_dim = 2
        
    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test
        
        
    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> tuple:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A tuple of data.
        """
        pass

#--------------------------------------------------------

class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems. Inherit time-dependent problems from here."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_transitions: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_transitions = allowed_transitions
        
    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0

        if self.fix_input_to_time_step is not None:
            assert (
                self.fix_input_to_time_step + self.max_num_time_steps
                <= self.max_num_time_steps
            )

            self.multiplier = self.max_num_time_steps
            #print(self.multiplier, "MULTI")
        else:
            if self.allowed_transitions is None:
                self.time_indices = []
                i = 0
                for j in range(i, self.max_num_time_steps + 1):
                    self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            else:
                self.time_indices = []
                for i in range(self.max_num_time_steps+1):
                    for j in range(i, self.max_num_time_steps + 1):
                        if (j-i) in self.allowed_transitions:
                            self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            
            self.multiplier = len(self.time_indices)
            print("time_indices", self.time_indices)
        
        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

#--------------------------------------------------------
# Navier-Stokes Datasets:
#--------------------------------------------------------

class BrownianBridgeTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/bm.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------

class VortexSheetTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/vortex_sheet.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------

class SinesTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/sin.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0

        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------

class PiecewiseConstantsTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/pwc.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")
        
        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0

        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------

class GaussiansTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/gauss.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")
        
        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .reshape(2, self.resolution, self.resolution)
        )
        
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label
    
#--------------------------------------------------------

class ComplicatedShearLayerTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 40000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            assert self.resolution in [64, 128]
            data_path_1 = self.data_path + "/data_ns/N" + str(self.resolution) + "_1.nc"
            data_path_2 = self.data_path + "/data_ns/N" + str(self.resolution) + "_2.nc"
            data_path_3 = self.data_path + "/data_ns/N" + str(self.resolution) + "_3.nc"
            data_path_4 = self.data_path + "/data_ns/N" + str(self.resolution) + "_4.nc"
        else:
            raise NotImplementedError()

        reader_1 = h5py.File(data_path_1, "r")
        reader_2 = h5py.File(data_path_2, "r")
        reader_3 = h5py.File(data_path_3, "r")
        reader_4 = h5py.File(data_path_4, "r")
        self.reader = [reader_1, reader_2, reader_3, reader_4]

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier
        
        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        #---------------------------
        if self.resolution == 128:
            i_fix = i + 10000
        else:
            i_fix = i
        
        if self.which == "train":
            which_reader = i // 10000
        else:
            which_reader = 3
        #---------------------------
        
        axes = (0,2,1)
        
        inputs = (
            torch.from_numpy(
                np.transpose(self.reader[which_reader]["sample_" + str(i_fix + self.start)][:][t1], axes = axes)
            )
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        label = (
            torch.from_numpy(
                np.transpose(self.reader[which_reader]["sample_" + str(i_fix + self.start)][:][t2], axes = axes)
            )
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label
    
#--------------------------------------------------------
# Compressible Euler Datasets:
#--------------------------------------------------------

class KelvinHelmholtzTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + '/kh.nc'
        else:
            raise NotImplementedError()

        file = nc.Dataset(data_path, "r")
        self.reader  = file["data"]
        
        self.mean = torch.tensor([0.80, 0.0,   0.0,   1.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.tensor(self.reader[i + self.start,t1,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.tensor(self.reader[i + self.start,t2,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label
    
#--------------------------------------------------------

class RiemannTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + '/riemann.nc'
        else:
            raise NotImplementedError()

        file = nc.Dataset(data_path, "r")
        self.reader  = file["data"]

        self.mean = torch.tensor([0.80, 0.0,   0.0,   0.215], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185],  dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.tensor(self.reader[i + self.start,t1,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.tensor(self.reader[i + self.start,t2,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label
    
#--------------------------------------------------------

class RiemannCurvedTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + '/riemann_curved.nc'
        else:
            raise NotImplementedError()

        file = nc.Dataset(data_path, "r")
        self.reader  = file["data"]

        self.mean = torch.tensor([0.80, 0.0,   0.0,   0.553], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185],  dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.tensor(self.reader[i + self.start,t1,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.tensor(self.reader[i + self.start,t2,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------

class EulerGaussTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + '/gauss.nc'
        else:
            raise NotImplementedError()

        file = nc.Dataset(data_path, "r")
        self.reader  = file["data"]
        
        self.mean = torch.tensor([0.80, 0.0,   0.0,   2.513], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.tensor(self.reader[i + self.start,t1,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.tensor(self.reader[i + self.start,t2,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------    

class RiemannKHTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + '/riemann_kh.nc'
        else:
            raise NotImplementedError()

        file = nc.Dataset(data_path, "r")
        self.reader  = file["data"]
        
        self.mean = torch.tensor([0.80, 0.0,     0.0,  1.33], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185],  dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.tensor(self.reader[i + self.start,t1,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.tensor(self.reader[i + self.start,t2,:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------
# Richtmyer-Meshkov Experiment:
#--------------------------------------------------------
class RichtmyerMeshkov(BaseTimeDataset):
    def __init__(self, *args, tracer = False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128
        self.tracer = tracer

        if self.in_dist:
            data_path = self.data_path +'/richtmyer_meshkov.nc'
        
        else:
            raise NotImplementedError()

        self.reader = nc.Dataset(data_path, "r")
        
        self.label_description = (
            "[rho],[u,v],[p]" if not tracer else "[rho],[u,v],[p],[tracer]"
        )
        
        self.constants = {
            "mean": torch.tensor(
                [1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036]
            ).unsqueeze(1).unsqueeze(1),
            "std": torch.tensor(
                [0.5543239, 0.24304213, 0.2430597, 0.89639103]
            ).unsqueeze(1).unsqueeze(1),
            "time": 20.0,
            "tracer_mean": 1.3658239,
            "tracer_std": 0.46400866,
        }
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        label = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]
        
        if self.tracer:
            input_tracer = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t1, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t2, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            inputs = torch.cat([inputs, input_tracer], dim=0)
            label = torch.cat([label, output_tracer], dim=0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label



#--------------------------------------------------------
# Rayleigh-Taylor Experiment (Euler + Force):
#--------------------------------------------------------

class RayleighTaylor(BaseTimeDataset):
    def __init__(self, *args, tracer=False,  **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 10

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128
        self.tracer = tracer
        
        if self.in_dist:
            data_path = self.data_path + '/rayleigh_taylor.nc'
        
        else:
            raise NotImplementedError()

        self.reader = nc.Dataset(data_path, "r")
        
        self.label_description = (
            "[rho],[u,v],[p],[g]" if not tracer else "[rho],[u,v],[p],[tracer],[g]"
        )
        
        self.constants = {
            "mean": torch.tensor(
               [0.8970493, 4.0316996e-13, -1.3858967e-13, 0.7133829, -1.7055787]
            ).unsqueeze(1).unsqueeze(1),
            "std": torch.tensor(
                [0.12857835, 0.014896976, 0.014896975, 0.21293919, 0.40131348]
            ).unsqueeze(1).unsqueeze(1),
            "time": 10.0,
            "tracer_mean": 1.8061695,
            "tracer_std": 0.37115487,
        }
        
        print(self.which, self.N_test, self.N_max - self.N_test, "CHECK")
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]
        
        
        inputs = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        g_1 = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        g_2 = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.constants["mean"][:4]) / self.constants["std"][:4]
        g_1 = (g_1 - self.constants["mean"][4]) / self.constants["std"][4]
        g_2 = (g_2 - self.constants["mean"][4]) / self.constants["std"][4]
        label = (label - self.constants["mean"][:4]) / self.constants["std"][:4]
        
        
        if self.tracer:
            tracer_1 = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t1, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            tracer_2 = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t2, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            tracer_1 = (tracer_1 - self.constants["tracer_mean"]) / self.constants[
                "tracer_std"
            ]
            tracer_2 = (tracer_2 - self.constants["tracer_mean"]) / self.constants[
                "tracer_std"
            ]
            inputs = torch.cat([inputs, tracer_1, g_1], dim=0)
            label = torch.cat([label, tracer_2, g_2], dim=0)
        else:
            inputs = torch.cat([inputs, g_1], dim=0)
            label = torch.cat([label, g_2], dim=0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------
# Allen-Cahn Equation:
#--------------------------------------------------------

class AllenCahn(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 19

        self.N_max = 15000
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/reaction_diffusion/allen_cahn.nc"
        self.reader = nc.Dataset(data_path, "r")

        self.constants = {
            "mean": 0.002484262,
            "std": 0.65351176,
            "time": 19.0,
        }

        self.input_dim = 1
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, labels

#--------------------------------------------------------
# Poisson Equation:
#--------------------------------------------------------

class PoissonBase(BaseDataset):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128

        self.file_path = os.path.join(self.data_path, file_path)
        self.reader = nc.Dataset(self.file_path, "r")
        self.constants = {
                        "mean_source": 0.014822142414492256,
                        "std_source": 4.755138816607612,
                        "mean_solution": 0.0005603458434937093,
                        "std_solution": 0.02401226126952699,
                        }

        self.input_dim = 1
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.reader.variables["source"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (
            torch.from_numpy(self.reader.variables["solution"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean_source"]) / self.constants["std_source"]
        labels = (labels - self.constants["mean_solution"]) / self.constants[
            "std_solution"
        ]
        
        return 1.0, inputs, labels

class PoissonGaussians(PoissonBase):
    def __init__(self, *args, **kwargs):
        # mean_source = 0.0608225485107185
        # std_source = 0.18010304094287755
        # mean_solution = 0.002326384792539932
        # std_solution = 0.05859948481241117
        super().__init__("poisson_equation/gaussians.nc", *args, **kwargs)


#--------------------------------------------------------
# Helmholts Equation:
#--------------------------------------------------------

class Helmholtz(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 19675
        self.N_val = 128
        self.N_test = 512
        self.resolution = 128

        self.file_path = os.path.join(
            self.data_path,
            "helmholtz/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5",
        )
        self.reader = h5py.File(self.file_path, "r")
        self.mean = 0.11523915668552
        self.std = 0.8279975746000605

        self.input_dim = 2
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.reader["Sample_" + str(idx + self.start)]["a"][:])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs = inputs - 1
        b = float(np.array(self.reader["Sample_" + str(idx + self.start)]["bc"]))
        bc = b * torch.ones_like(inputs)
        inputs = torch.cat((inputs, bc), dim=0)

        labels = (
            torch.from_numpy(self.reader["Sample_" + str(idx + self.start)]["u"][:])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (labels - self.mean) / self.std

        return 1.0, inputs, labels

#--------------------------------------------------------
# Airfoil Dataset (Steady):
#--------------------------------------------------------

class Airfoil(BaseDataset):
    def __init__(self, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10869
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/compressible_flow/steady/airfoil.nc"
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.92984116,
            "std": 0.10864315,
        }

        self.input_dim = 1
        self.label_description = "[rho]"

        self.post_init()

    def __getitem__(self, idx):
        
        time = 1.0

        inputs = (
            torch.from_numpy(self.reader["solution"][idx + self.start, 0])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][idx + self.start, 1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, labels
    
#--------------------------------------------------------
# Wave Equation:
#--------------------------------------------------------

class WaveSeismic(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/wave_equation/seismic_20step.nc"
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.03467443221585092,
            "std": 0.10442421752963911,
            "mean_c": 3498.5644380917424,
            "std_c": 647.843958567462,
            "time": 20.0,
        }

        self.input_dim = 2
        self.label_description = "[u],[c]"

        self.post_init()

    def __getitem__(self, idx):
        
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]
        

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
            
        inputs = torch.cat([inputs, inputs_c], dim=0)
        labels = torch.cat([labels, inputs_c], dim=0)
        
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, labels

class WaveGaussians(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 15

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/wave_equation/gaussians_15step.nc"
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.0334376316,
            "std": 0.1171879068,
            "mean_c": 2618.4593933,
            "std_c": 601.51658913,
            "time": 15.0,
        }

        self.input_dim = 2
        self.label_description = "[u],[c]"

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]
        
        

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, inputs_c], dim=0)
        labels = torch.cat([labels, inputs_c], dim=0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, labels


#--------------------------------------------------------
# Kolmogorov:
#--------------------------------------------------------
    
class KolmogorovFlow(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/incompressible_fluids/forcing/kolmogorov_pwc.nc"
        self.reader = h5py.File(data_path, "r")

        self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.22, 0.22],  dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std_forcing = 0.0707
        
        
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.forcing = self.forcing / self.std_forcing
        
        self.input_dim = 3
        self.label_description = "[u,v],[f]"
    
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        

        inputs_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        inputs_v = (inputs_v - self.mean) / self.std
        label_v = (label_v - self.mean) / self.std
        
        inputs  = torch.cat((inputs_v, self.forcing), 0)
        label_v = torch.cat((label_v, self.forcing), 0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        
        
        if self.masked_input is not None:
            return time, inputs, label_v, self.mask
        else:
            return time, inputs, label_v
        
#-------------------------
# Navier-Stokes Tracers:
#-------------------------

class PiecewiseConstantsTraceTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/pwc_tracer.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")
        #0.391, 0.356
        #0.49198706571149564, 0.36194905497513363
        self.mean = torch.tensor([0,0,0.19586183], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor([0.391, 0.356,0.37], dtype=torch.float32).unsqueeze(1).unsqueeze(1)


        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0

        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(3, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(3, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, label
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from CNOModule import CNO

import h5py
import re
import einops
from typing import Sequence
import os
import numpy as onp
from jax import random

standard_NS_directory = '/scratch/PDEDatasets/pdearena/NavierStokes2D_smoke/'


class StandardNavierStokesDataset(Dataset):
    def __init__(self, directory : str, mode : str, keys :Sequence[str], previous_t_steps : int, prediction_t_steps : int):
        super().__init__()
        self.directory = directory
        file_names = []
        for f in os.listdir(directory):
            b = re.search(f'^NavierStokes2D_{mode}', f)
            if b is not None:
                file_names.append(directory + b.string)
        self.file_names = file_names
        self.previous_t_steps = previous_t_steps
        self.prediction_t_steps = prediction_t_steps
        self.mode = mode
        self.keys = keys


        self.files = files = [h5py.File(fn, 'r') for fn in self.file_names]

        file_index = []
        batch_index = []
        time_index = []
        for i in range(len(files)):
            s = files[i][self.mode][self.keys[0]].shape
            T = (s[1] - self.previous_t_steps - self.prediction_t_steps)
            if T > 0:
                file_index += list(i * onp.ones((s[0] * T)).astype(int))
                batch_index += list(einops.repeat(onp.arange(s[0]), 'B -> (B T )', T = T))
                time_index += list(einops.repeat(onp.arange(T), 'T -> (B T)', B = s[0]))
        
        self.file_index = file_index
        self.batch_index = batch_index
        self.time_index = time_index


    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, index):
        f = self.files[self.file_index[index]][self.mode]
        inputs = dict()
        outputs = dict()

        start_time_step = self.time_index[index]
        end_time_step = start_time_step + self.previous_t_steps
        prediction_time_step = end_time_step + self.prediction_t_steps
        

        for k in self.keys:
            inputs[k] = torch.tensor(f[k][self.batch_index[index], start_time_step:end_time_step])
            outputs[k] = torch.tensor(f[k][self.batch_index[index], end_time_step:prediction_time_step])

        inputs = torch.cat(list(inputs.values()), 0)
        outputs = torch.cat(list(outputs.values()), 0)

        return inputs.float(), outputs.float()

def default_param(network_properties):
    
    if "channel_multiplier" not in network_properties:
        network_properties["channel_multiplier"] = 32
    
    if "half_width_mult" not in network_properties:
        network_properties["half_width_mult"] = 1
    
    if "lrelu_upsampling" not in network_properties:
        network_properties["lrelu_upsampling"] = 2

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
    if "out_size" not in network_properties:
        network_properties["out_size"] = 1
    
    if "radial" not in network_properties:
        network_properties["radial_filter"] = 0
    
    if "cutoff_den" not in network_properties:
        network_properties["cutoff_den"] = 2.0001
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
         network_properties["retrain"] = 4
    
    if "kernel_size" not in network_properties:
        network_properties["kernel_size"] = 3
    
    if "activation" not in network_properties:
        network_properties["activation"] = 'cno_lrelu'
    
    return network_properties

class StandardNavierStokes:
    def __init__(self, network_properties, device, batch_size, training_samples, size):

        #Must have parameters: ------------------------------------------------        

        self.in_size = 128
        previous_step = 4
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = previous_step * 3 + 2*self.N_Fourier_F,     # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation,
                        out_dim = 3
                        ).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        
        num_workers = 0

        train_dataset = StandardNavierStokesDataset(standard_NS_directory, 'train', ['u', 'vx', 'vy'], 4, 1)
        train_dataset = Subset(train_dataset, onp.arange(training_samples))

        validation_dataset = StandardNavierStokesDataset(standard_NS_directory, 'valid', ['u', 'vx', 'vy'], 4, 1)
        validation_dataset = Subset(validation_dataset, onp.arange(128))

        test_dataset = StandardNavierStokesDataset(standard_NS_directory, 'test', ['u', 'vx', 'vy'], 4, 1)
        test_dataset = Subset(test_dataset, onp.arange(128))

        self.train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.val_loader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=False, num_workers=num_workers)
        




if __name__ == "__main__":
    
    ns_dataset = StandardNavierStokesDataset(standard_NS_directory, 'train', 
                                                        ['u', 'vx', 'vy'], 
                                                        5, 
                                                        1)
    ns_loader = DataLoader(ns_dataset, 64)

    for i, o in ns_loader:
        print(i.shape, o.shape)
        break
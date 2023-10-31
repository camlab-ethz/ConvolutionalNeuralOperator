import random
import os
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from WaveletResNetModule import WaveResNet
from torch.utils.data import Dataset

import netCDF4
import scipy
from scipy import ndimage


from training.FourierFeatures import FourierFeatures

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

#------------------------------------------------------------------------------
#Navier-Stokes data:
    
class ShearLayerDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        self.s = s
        #The file:
        
        #in_dist= False    
        
        if in_dist:
            #folder = "data/Euler_Shear/" #In-distribution file
            #sself.folder = "data/Euler_Shear_10000/ddsl_N128/" #In-distribution file
            
            if self.s==64:
                #self.file_data = "/cluster/scratch/braonic/data/NavierStokes64_1024.h5"
                self.file_data  = "data/NavierStokes64_1024.h5"
                self.reader = h5py.File(self.file_data, 'r') 
                self.N_max = 1024
            else:
                self.folder = "data/Euler_Shear/ddsl_N128/" #In-distribution file
                self.N_max = 1024
        else:
            self.folder = "data/Euler_Shear_0_05/ddsl_N128/" #Out-of_-distribution file
            self.N_max = 1024
        
        self.n_val  = 128
        self.n_test = 128
        
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model= -2.0383243560791016
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max  - self.n_test
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        if self.s == 64:
            
            inp = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(self.s, self.s)
            
            inp = ndimage.gaussian_filter(inp, sigma = 2)
            
            
            #inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            inputs = torch.from_numpy(inp).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:
            
            file_input  = self.folder + "sample_" + str(index + self.start) + "_time_0.nc" 
            file_output = self.folder + "sample_" + str(index + self.start) + "_time_10.nc" 
            
            f = netCDF4.Dataset(file_input,'r')
            if self.s<128:
                inputs = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.s).reshape(1, self.s, self.s)
                
            else:
                inputs = np.array(f.variables['u'][:]).reshape(1, self.s, self.s)  
            f.close()
            
            f = netCDF4.Dataset(file_output,'r')
            if self.s<128:
                labels = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.s).reshape(1, self.s, self.s)
            else:
                labels = np.array(f.variables['u'][:]).reshape(1, self.s, self.s)
            
            
            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

            f.close()

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        #inputs = inputs + 0.05*torch.randn_like(inputs)
        
        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, file = "ddsl_N128/", in_dist = True):

        #Must have parameters: ------------------------------------------------        

        """
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
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
        """
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        num_blocks = network_properties["num_blocks"]
        channels = network_properties["channels"]
        retrain = network_properties["retrain"]
        
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        
        
        self.model = WaveResNet(in_channels = 1,
                                out_channels = 1,
                                latent_channels = channels,
                                num_blocks = num_blocks,
                                conv_kernel = kernel_size,
                                batch_norm = True,
                                device = device
                                ).to(device)

        #----------------------------------------------------------------------
        
        

        #Change number of workers accoirding to your preference
        num_workers = 16
        s = 64
        self.train_loader = DataLoader(ShearLayerDataset("training", 0, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(ShearLayerDataset("validation", 0, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=8)

        #val_data = ShearLayerDataset("validation", self.N_Fourier_F, training_samples, s)
        self.test_loader = DataLoader(ShearLayerDataset("test", 0, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

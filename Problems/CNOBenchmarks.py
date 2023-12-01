import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from CNOModule import CNO
from torch.utils.data import Dataset

import scipy

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

#Load default parameters:
    
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

#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

#------------------------------------------------------------------------------
#Navier-Stokes data:
#   From 0 to 750 : training samples (750)
#   From 1024 - 128 - 128 to 1024 - 128 : validation samples (128)
#   From 1024 - 128 to 1024 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class ShearLayerDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 750, s=64, in_dist = True):
        
        self.s = s
        self.in_dist = in_dist
        #The file:
        
        if in_dist:
            if self.s==64:
                self.file_data = "data/NavierStokes_64x64_IN.h5" #In-distribution file 64x64               
            else:
                self.file_data = "data/NavierStokes_128x128_IN.h5"   #In-distribution file 128x128
        else:
            self.file_data = "data/NavierStokes_128x128_OUT.h5"  #Out-of_-distribution file 128x128
        
        self.reader = h5py.File(self.file_data, 'r') 
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
        
        if self.s == 64 and self.in_dist:
            inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:
            
            inputs = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(1,1,self.s, self.s)
            labels = self.reader['Sample_' + str(index + self.start)]["output"][:].reshape(1,1, self.s, self.s)
            
            if self.s<128:
                inputs = downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)
            
            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

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
    def __init__(self, network_properties, device, batch_size, training_samples, size, in_dist = True):

        #Must have parameters: ------------------------------------------------        

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
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,     # Number of input channels.
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
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        
        num_workers = 0
        self.train_loader = DataLoader(ShearLayerDataset("training", self.N_Fourier_F, training_samples, size), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ShearLayerDataset("validation", self.N_Fourier_F, training_samples, size), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ShearLayerDataset("test", self.N_Fourier_F, training_samples, size, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Poisson data:
#   From 0 to 1024 : training samples (1024)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        #Load normalization constants from the TRAINING set:
        file_data_train = "data/PoissonData_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        
        #The file:
        if in_dist:
            self.file_data = "data/PoissonData_64x64_IN.h5"
        else:
            self.file_data = "data/PoissonData_64x64_OUT.h5"

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024+128
            else:
                self.length = 256
                self.start = 0 
        
        #Load different resolutions
        if s!=64:
            self.file_data = "data/PoissonData_NEW_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class SinFrequency:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
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
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)

        #----------------------------------------------------------------------
        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(SinFrequencyDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(SinFrequencyDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(SinFrequencyDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
# Wave data
#   From 0 to 512 : training samples (512)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, t = 5, s = 64, in_dist = True):
        
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        #Load normalization constants from the TRAINING set:
        file_data_train = "data/WaveData_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #Default file:       
        if in_dist:
            self.file_data = "data/WaveData_64x64_IN.h5"
        else:
            self.file_data = "data/WaveData_64x64_OUT.h5"
        
        #What time? DEFAULT : t = 5
        self.t = t
                        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0
        
        self.s = s
        if s!=64:
            self.file_data = "data/WaveData_24modes_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        grid = torch.zeros((self.s, self.s,2))

        for i in range(self.s):
            for j in range(self.s):
                grid[i, j][0] = i/(self.s - 1)
                grid[i, j][1] = j/(self.s - 1)
                
        return grid


class WaveEquation:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
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
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 0
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Allen-Cahn data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class AllenCahnDataset(Dataset):
    def __init__(self, which="training", nf = 0, training_samples = 256, s=64, in_dist = True):

        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = "data/AllenCahn_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #Default file:
        if in_dist:
            self.file_data = "data/AllenCahn_64x64_IN.h5"
        else:            
            self.file_data = "data/AllenCahn_64x64_OUT.h5"
        self.reader = h5py.File(self.file_data, 'r')
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 256
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 256 + 128
            else:
                self.length = 128
                self.start = 0 
        
        #Default:
        self.N_Fourier_F = nf
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        #print(inputs.shape)            
        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class AllenCahn:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024,  s = 64, in_dist = True):
        
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
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)

        #----------------------------------------------------------------------
        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AllenCahnDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AllenCahnDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(AllenCahnDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Smooth Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class ContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized        
        #Default file:       
        if in_dist:
            self.file_data = "data/ContTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/ContTranslation_64x64_OUT.h5"
        
        self.reader = h5py.File(self.file_data, 'r') 

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512 + 256
            else:
                self.length = 256
                self.start = 0

        #Default:
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class ContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):

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
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(ContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ContTranslationDataset("test", self.N_Fourier_F, training_samples, s,in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Discontinuous Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class DiscContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        
        if in_dist:
            self.file_data = "data/DiscTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/DiscTranslation_64x64_OUT.h5"
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512+256
            else:
                self.length = 256
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class DiscContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 64, in_dist = True):
        
        
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

        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]

        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]

        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)


        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DiscContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DiscContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(DiscContTranslationDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Compressible Euler data
#   From 0 to 750 : training samples (750)
#   From 750 to 750 + 128 : validation samples (128)
#   From 750 + 128 to 750 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class AirfoilDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 128, in_dist = True):
        
        #We DO NOT normalize the data in this case
        
        if in_dist:
            self.file_data = "data/Airfoil_128x128_IN.h5"
        else:
            self.file_data = "data/Airfoil_128x128_OUT.h5"
        
        #in_dist = False
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 128
            self.start = 750
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 750 + 128
            else:
                self.length = 128
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 128, 128)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 128, 128)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 128)
        y = torch.linspace(0, 1, 128)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class Airfoil:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 128, in_dist = True):
        #Must have parameters: ------------------------------------------------        

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
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
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
                        activation = activation).to(device)
        
        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AirfoilDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AirfoilDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AirfoilDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Darcy Flow data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class DarcyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=256, insample=True):
        
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = "data/Darcy_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        if insample:
            self.file_data = "data/Darcy_64x64_IN.h5"
        else:
            self.file_data = "data/Darcy_64x64_OUT.h5"
    
        self.reader = h5py.File(self.file_data, 'r')
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "testing":
            if insample:
                self.length = 128
                self.start = training_samples + 128
            else:
                self.length = 128
                self.start = 0

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid
    
class Darcy:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):
        
        #Must have parameters: ------------------------------------------------        

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

        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------

        torch.manual_seed(retrain)

        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                 # Input spatial size
                        N_layers = N_layers,                    # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                          # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DarcyDataset("training", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DarcyDataset("validation", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(DarcyDataset("testing", self.N_Fourier_F, training_samples, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

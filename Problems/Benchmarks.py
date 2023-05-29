import random

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ModelModule import ContConv2D
from torch.utils.data import Dataset

import netCDF4
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
    
    if "res_len" not in network_properties:
        network_properties["res_len"] = 1

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
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
    
    return network_properties

#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

#------------------------------------------------------------------------------
#Navier-Stokes data:
    
class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, file = "ddsl_N128/", in_dist = True):

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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
                
        if in_dist:
            folder = "data/Euler_Shear/" #In-distribution file
        else:
            folder = "data/Euler_Shear_0_05/" #Out-of_-distribution file
        

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #----------------------------------------------------------------------

        samples_inputs, samples_outputs = self.get_data(folder + file, 1024)
        
        #Normalization constants:
        m = 1.4294605255126953
        M = -1.4294605255126953
        samples_inputs  = self.normalize(samples_inputs, m, M)
        m = 2.0602376461029053
        M = -2.0383081436157227
        samples_outputs = self.normalize(samples_outputs, m, M)

        #Training samples
        training_inputs = samples_inputs[:training_samples]
        training_outputs = samples_outputs[:training_samples]

        #Validation samples
        val_inputs = samples_inputs[896:] 
        val_outputs = samples_outputs[896:]
        
        #Test samples:
        testing_inputs = samples_inputs[750:750+128] 
        testing_outputs = samples_outputs[750:750+128]

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(val_inputs, val_outputs), batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)


    def normalize(self, data, m, M):
        return (data - m)/(M - m)

    def get_data(self, folder, n_samples):
        
        input_data = np.zeros((n_samples, 1 + 2*self.N_Fourier_F, self.in_size, self.in_size))
        output_data = np.zeros((n_samples, 1, self.in_size, self.in_size))
        
        grid = self.get_grid()
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
        
        if self.N_Fourier_F>0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2,0,1)
            ff_grid = ff_grid.detach().cpu().numpy()
        
        for i in range(n_samples):
            
            file_input  = folder + "sample_" + str(i) + "_time_0.nc" 
            file_output = folder + "sample_" + str(i) + "_time_10.nc" 

            f = netCDF4.Dataset(file_input,'r')
            if self.in_size<128:
                input_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
            else:
                input_data[i, 0] = np.array(f.variables['u'][:])
            f.close()

            f = netCDF4.Dataset(file_output,'r')
            if self.in_size<128:
                output_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
            else:
                output_data[i, 0] = np.array(f.variables['u'][:])
            f.close()
            
            if self.N_Fourier_F>0:
                input_data[i,1:,:,:] = ff_grid
                
        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        grid = torch.zeros((self.in_size, self.in_size,2))
        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i/(self.in_size - 1)
                grid[i, j][1] = j/(self.in_size - 1)
                
        return grid

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Poisson data:

class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        
        #The file:
        self.file_data = "data/PoissonData_IN_TRAINING.h5"
        
        #Load normalization constants from the TRAINING set:
        self.reader = h5py.File(self.file_data, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist: #Is it in-distribution?
                self.length = 256
                self.start = 1024+128
            else:
                self.file_data = "data/PoissonData_OUT2_20modes.h5"
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
        
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=s,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 16

        self.train_loader = DataLoader(SinFrequencyDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(SinFrequencyDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(SinFrequencyDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
      
#Wave data

class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, t = 5, s = 64, in_dist = True):
        
        #Default file:
        self.file_data = "data/WaveData_IN_24modes.h5"
        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #What time?
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
                self.file_data = "data/WaveData_OUT_32modes_085decay.h5"
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
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=s,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 16
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Allen-Cahn data
  
class AllenCahnDataset(Dataset):
    def __init__(self, which="training", nf = 0, training_samples = 1024, s=64, in_dist = True):
        
        
        #Default file:
        self.file_data = "data/AllenCahn_NEW.h5"
        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
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
                self.file_data = "data/AllenCahn_OUT_16modes_random_decay_085_115.h5"
                self.length = 128
                self.start = 0

        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
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
        
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=s,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 16

        self.train_loader = DataLoader(AllenCahnDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AllenCahnDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AllenCahnDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  
#Smooth Transport data

class ContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        #The in-distribution and out-of-distribution data is a bit mixed up, so you should use the code below to load it properly
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            self.file_data = "data/ContTranslation.h5"
        elif which == "validation":
             self.length = 256
             self.start = 0
             self.file_data = "data/ContTranslation_test_in_sample.h5"
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 256
                self.file_data = "data/ContTranslation_test_in_sample.h5" 
            else:
                self.length = 256
                self.start = 512+256
                self.file_data = "data/ContTranslation.h5"

        self.reader = h5py.File(self.file_data, 'r') 
        
        #Default:
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        #print("I AM HERE BRE")
        
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
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=s,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 16

        self.train_loader = DataLoader(ContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ContTranslationDataset("test", self.N_Fourier_F, training_samples, s,in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Discontinuous Transport data

class DiscContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        #The in-distribution and out-of-distribution data is a bit mixed up, so you should use the code below to load it properly
        
        self.file_data = "data/DiscTranslation.h5"
        
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
                self.start = 1024

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
       
        #Must have parameters: ------------------------------------------------        

       if "in_size" in network_properties:
           self.in_size = network_properties["in_size"]
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
       
       
       #Load default parameters if they are not in network_properties
       network_properties = default_param(network_properties)
       
       
       kernel_size = network_properties["kernel_size"]
       channel_multiplier = network_properties["channel_multiplier"]
       res_len = network_properties["res_len"]
       retrain = network_properties["retrain"]
       self.N_Fourier_F = network_properties["FourierF"]
       
       #Filter properties: ---------------------------------------------------
       cutoff_den = network_properties["cutoff_den"]
       filter_size = network_properties["filter_size"]
       radial = network_properties["radial_filter"]
       half_width_mult = network_properties["half_width_mult"]
       lrelu_upsampling = network_properties["lrelu_upsampling"]
   
       torch.manual_seed(retrain)
       
       #----------------------------------------------------------------------

       self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                               in_size=s,
                               cutoff_den=cutoff_den,
                               N_layers=N_layers,
                               N_res=N_res,
                               radial=radial,
                               filter_size=filter_size,
                               conv_kernel=kernel_size,
                               lrelu_upsampling = lrelu_upsampling,
                               half_width_mult = half_width_mult,
                               channel_multiplier = channel_multiplier,
                               length_res = res_len
                               ).to(device)

       #Change number of workers accoirding to your preference
       num_workers = 16

       self.train_loader = DataLoader(DiscContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
       self.val_loader = DataLoader(DiscContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
       self.test_loader = DataLoader(DiscContTranslationDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Compressible Euler data
      
class AirfoilDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 128, in_dist = True):
        
        #We DO NOT normalize the data in this case
        
        self.file_data = "data/Airfoil.h5"
        
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
                self.file_data = "data/Airfoil_out.h5"

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        #Here, we "cut" the part of the shape that we are intersted (to get 128 x 128 resolution)
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32)[90:90+256, 128:128+256][::2,::2].reshape(1, 128, 128)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32)[90:90+256, 128:128+256][::2,::2].reshape(1, 128, 128)
        
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
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = ContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=s,
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                length_res = res_len
                                ).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 16

        self.train_loader = DataLoader(AirfoilDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AirfoilDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AirfoilDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

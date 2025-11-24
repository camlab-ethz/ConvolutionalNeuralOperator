import random

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from UModelModule import UContConv2D
from ModelModule import ContConv2D

from utilities_nemo.data_loader import load_input_output, load_bathymetry
from utilities_nemo.train_util import normalize_encode, normalize_bath_encode
from torch.utils.data import Dataset

import scipy

from CNO_Processing.FourierFeatures import FourierFeatures

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class DarcyFlow:
    def __init__(self, network_properties, device, batch_size, training_samples):

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)

        inputs, outputs = self.get_data(2000)
        training_inputs, training_outputs = inputs[:training_samples], outputs[:training_samples]
        testing_inputs, testing_outputs = inputs[1000:], outputs[1000:]


        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def get_data(self, n_samples):
        input_data = np.zeros((n_samples, 1 + 2*self.N_Fourier_F, 32, 32))
        output_data = np.zeros((n_samples, 1, 32, 32))
        
        with h5py.File("data/PoissonData.h5", "r") as f:
            grid = self.get_grid()
                        
            if self.N_Fourier_F>0:
                FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            
            if self.N_Fourier_F>0:
                ff_grid = FF(grid)
                ff_grid = ff_grid.permute(2,0,1)
                ff_grid = ff_grid.detach().cpu().numpy()
            
            for i in range(2000):
                name = "sample_" + str(i)
                input = f[name]["input"][:]
                output = f[name]["output"][:]
                # input_data[i] =np.concatenate((grid,input.reshape(1, 32, 32)),0)
                input_data[i][0] = input
                
                if self.N_Fourier_F>0:
                    input_data[i,1:] = ff_grid
                
                output_data[i] = output.reshape(1, 32, 32)

            return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        with h5py.File("data/PoissonData.h5", "r") as f:
            grid = torch.tensor(f["grid"][:]).type(torch.float32)
        return grid


#------------------------------------------------------------------------------------------------------------------------------------

class NavierStokes_VIDON:
    def __init__(self, network_properties, device, batch_size,training_samples):

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        
        torch.manual_seed(retrain)

        training_inputs, training_outputs = self.get_data("data/data_32_32__Navier_Stokes/training_data_grid_time_5.hdf5", 1000)
        testing_inputs = training_inputs[500:]
        testing_outputs = training_outputs[500:]
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]
                        
        training_inputs, testing_inputs = self.normalize(training_inputs, testing_inputs)
        training_outputs, testing_outputs = self.normalize(training_outputs, testing_outputs)



        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data, data_t):
        M = max(torch.max(data).item(), torch.max(data_t).item())
        m = min(torch.min(data).item(), torch.min(data_t).item())

        return (data - m) / (M - m), (data_t - m) / (M - m)

    def get_data(self, file, n_samples):
        input_data = np.zeros((n_samples, 33, 33, 1 + 2*self.N_Fourier_F))
        output_data = np.zeros((n_samples, 33, 33, 1))
        
        grid = self.get_grid()
        
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
        
        if self.N_Fourier_F>0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.detach().cpu().numpy()
        
        with h5py.File(file, "r") as f:
            a_group_key = list(f.keys())

            for i in range(n_samples):
                s_in = "input_" + str(i)
                s_out = "output_" + str(i)

                input_data[i,:,:,0] = torch.tensor(np.array(f[a_group_key[0]][s_in])).reshape(33,33)
                
                if self.N_Fourier_F>0:
                    input_data[i,:,:,1:] = ff_grid
               
                output_data[i,:,:,0] = torch.tensor(np.array(f[a_group_key[1]][s_out])).reshape(33,33)

            return torch.tensor(input_data).type(torch.float32).permute(0, 3, 1, 2), torch.tensor(output_data).type(torch.float32).permute(0, 3, 1, 2)

    def get_grid(self):
        grid = torch.zeros((33, 33,2))

        for i in range(33):
            for j in range(33):
                grid[i, j][0] = i/32.0
                grid[i, j][1] = j/32.0
        return grid

#------------------------------------------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------------------------------------------


import netCDF4


class ShearLayer64:
    def __init__(self, network_properties, device, batch_size, training_samples):

        self.in_size = network_properties["in_size"]
        
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        
        if "half_width_mult" in network_properties:
            half_width_mult = network_properties["half_width_mult"]
        else:
            half_width_mult = 1
        
        self.N_Fourier_F = network_properties["FourierF"]
        
        
        torch.manual_seed(retrain)

        training_inputs, training_outputs = self.get_data("/cluster/scratch/braonic/Euler_Shear/ddsl_N64/", 1024)
        #training_inputs, training_outputs = self.get_data("data/Euler_Shear/ddsl_N64/", 1024)
        
        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        testing_inputs = training_inputs[896:] 
        testing_outputs = training_outputs[896:]
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]


        self.model = ContConv2D(in_channels = 1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                half_width_mult = half_width_mult
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data):
        m = torch.max(data)
        M = torch.min(data)
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
            if self.in_size<64:
                input_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,64,64), self.in_size)
            else:
                input_data[i, 0] = np.array(f.variables['u'][:])
            f.close()

            f = netCDF4.Dataset(file_output,'r')
            if self.in_size<64:
                output_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,64,64), self.in_size)
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


#------------------------------------------------------------------------------------------------------------------------------------

class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, file = "ddsl_N128/", cluster = True):

        self.in_size = network_properties["in_size"]

        assert self.in_size<=128
        
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)

        if cluster:
            training_inputs, training_outputs = self.get_data("/cluster/scratch/braonic/Euler_Shear/"+file, 1024, cluster)
        else:
            training_inputs, training_outputs = self.get_data("data/Euler_Shear/"+file, 1024, cluster)
        
        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        testing_inputs = training_inputs[896:] 
        testing_outputs = training_outputs[896:]
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]


        self.model = ContConv2D(in_channels = 1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data):
        m = torch.max(data)
        M = torch.min(data)
        return (data - m)/(M - m)

    def get_data(self, folder, n_samples, cluster):
        
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
            
            if i>=896:
                if cluster:
                    folder = "/cluster/scratch/braonic/Euler_Shear/ddsl_N128/"
                else:
                    folder = "data/Euler_Shear/ddsl_N128/"
          
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

#------------------------------------------------------------------------------------------------------------------------------------

class ShearLayerRecursive:
    def __init__(self, network_properties, device, batch_size, training_samples):

        #The sequence is at times : 0, 1, 2, ..., 9, 10
        #We try to predict 0 -> 2 -> 4 ... -> 8 -> 10        

        self.in_size = network_properties["in_size"]

        assert self.in_size<=128
        
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)

        # We try recursive CNO on 128 x 128 grid
        # We have n_samples training samples and 64 test samples
        # We will
        
        #training_inputs, training_outputs = self.get_data("/cluster/scratch/braonic/Euler_Shear/ddsl_N128/", training_samples)
        training_inputs, training_outputs, self.rec_data = self.get_data("data/Euler_Shear/ddsl_N128/", training_samples)
        
        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        self.rec_data    = self.normalize(self.rec_data)
        self.ff          = training_inputs[0, 1:,:,:]
        
        testing_inputs = training_inputs[5*training_samples:5*(training_samples+64)] 
        testing_outputs = training_outputs[5*training_samples:5*(training_samples+64)]
        training_inputs = training_inputs[:5*training_samples]
        training_outputs = training_outputs[:5*training_samples]

        self.model = ContConv2D(in_channels = 1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)
        
        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data):
        m = torch.max(data)
        M = torch.min(data)
        return (data - m)/(M - m)

    def get_data(self, folder, n_samples):
        input_data = np.zeros((5*(n_samples + 64), 1 + 2*self.N_Fourier_F, self.in_size, self.in_size))
        output_data = np.zeros((5*(n_samples + 64), 1, self.in_size, self.in_size))
        
        grid = self.get_grid()
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
        
        if self.N_Fourier_F>0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2,0,1)
            ff_grid = ff_grid.detach().cpu().numpy()
        
        cnt = 0
        
        a = np.arange(n_samples)
        b = np.arange(896, 896 + 64)
        c = np.append(a, b)
        for i in c:
            
            for t in range(0, 9, 2):
                
                file_input  = folder + "sample_" + str(i) + "_time_" + str(t) + ".nc" 
                file_output = folder + "sample_" + str(i) + "_time_" + str(t+2) + ".nc" 
                
                f = netCDF4.Dataset(file_input,'r')
                if self.in_size<128:
                    input_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    input_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                f = netCDF4.Dataset(file_output,'r')
                if self.in_size<128:
                    output_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    output_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                if self.N_Fourier_F>0:
                    input_data[cnt,1:,:,:] = ff_grid
            
                cnt+=1
            
        rec_data = np.zeros((64, 6, self.in_size, self.in_size))  
        cnt = 0
        
        for i in range(896 + 64, 896 + 2*64):    
            for t in range(0, 11, 2):
                
                file  = folder + "sample_" + str(i) + "_time_" + str(t) + ".nc" 
                
                f = netCDF4.Dataset(file,'r')
                if self.in_size<128:
                    rec_data[cnt, t//2] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    rec_data[cnt, t//2] = np.array(f.variables['u'][:])
                f.close()
            
            cnt+=1
            
        print(input_data[-1,0,0,0])
        print(output_data[-1,0,0,0])
        print(rec_data[-1,0,0,0])
        
        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32), torch.tensor(rec_data).type(torch.float32)
    
    def get_grid(self):
        
        grid = torch.zeros((self.in_size, self.in_size,2))

        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i/(self.in_size - 1)
                grid[i, j][1] = j/(self.in_size - 1)
                
        return grid

#------------------------------------------------------------------------------------------------------------------------------------

class ShearLayerRecursiveOut:
    def __init__(self, network_properties, device, batch_size, training_samples):

        #The sequence is at times : 0, 1, 2, ..., 9, 10
        #We try to predict 0 -> 2 -> 4
        #We want to test on 4 -> 6 -> 8 -> 10

        self.in_size = network_properties["in_size"]

        assert self.in_size<=128
        
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)

        # We try recursive CNO on 128 x 128 grid
        # We have n_samples training samples and 64 test samples
        # We will
        
        #training_inputs, training_outputs = self.get_data("/cluster/scratch/braonic/Euler_Shear/ddsl_N128/", training_samples)
        training_inputs, training_outputs = self.get_data("data/Euler_Shear/ddsl_N128/", training_samples)
        
        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        self.rec_data    = []
        self.ff          = training_inputs[0, 1:,:,:]
        
        testing_inputs = training_inputs[2*training_samples:] 
        testing_outputs = training_outputs[2*training_samples:]
        training_inputs = training_inputs[:2*training_samples]
        training_outputs = training_outputs[:2*training_samples]

        print(training_samples)

        self.model = ContConv2D(in_channels = 1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)
        
        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)
        
        print("DONE")
        
    def normalize(self, data):
        m = torch.max(data)
        M = torch.min(data)
        return (data - m)/(M - m)

    def get_data(self, folder, n_samples):
        input_data = np.zeros((2*n_samples + 5*64, 1 + 2*self.N_Fourier_F, self.in_size, self.in_size))
        output_data = np.zeros((2*n_samples + 5*64, 1, self.in_size, self.in_size))
        
        grid = self.get_grid()
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
        
        if self.N_Fourier_F>0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2,0,1)
            ff_grid = ff_grid.detach().cpu().numpy()
        
        cnt = 0
        
        a = np.arange(n_samples)
        b = np.arange(896, 896 + 64)
        
        for i in a:
            
            for t in range(0, 3, 2):
                
                file_input  = folder + "sample_" + str(i) + "_time_" + str(t) + ".nc" 
                file_output = folder + "sample_" + str(i) + "_time_" + str(t+2) + ".nc" 
                
                f = netCDF4.Dataset(file_input,'r')
                if self.in_size<128:
                    input_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    input_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                f = netCDF4.Dataset(file_output,'r')
                if self.in_size<128:
                    output_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    output_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                if self.N_Fourier_F>0:
                    input_data[cnt,1:,:,:] = ff_grid
                cnt+=1
            
        
        for i in b:
            
            for t in range(0, 9, 2):
                
                file_input  = folder + "sample_" + str(i) + "_time_" + str(t) + ".nc" 
                file_output = folder + "sample_" + str(i) + "_time_" + str(t+2) + ".nc" 
                
                f = netCDF4.Dataset(file_input,'r')
                if self.in_size<128:
                    input_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    input_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                f = netCDF4.Dataset(file_output,'r')
                if self.in_size<128:
                    output_data[cnt, 0] = downsample(np.array(f.variables['u'][:]).reshape(1,1,128,128), self.in_size)
                else:
                    output_data[cnt, 0] = np.array(f.variables['u'][:])
                f.close()
                
                if self.N_Fourier_F>0:
                    input_data[cnt,1:,:,:] = ff_grid
                cnt+=1
            
        
        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)
    
    def get_grid(self):
        
        grid = torch.zeros((self.in_size, self.in_size,2))

        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i/(self.in_size - 1)
                grid[i, j][1] = j/(self.in_size - 1)
                
        return grid



#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

#Equivarianve benchmarks:

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------


class Translation:
    def __init__(self, network_properties, device, batch_size):
        # inputs, outputs = self.get_data(2000)

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        
        torch.manual_seed(retrain)

        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        file_train = "data/RotationTranslation/TrainingData_RotationTranslation.h5"
        file_test = "data/RotationTranslation/TestData_RotationTranslation.h5"
        
        training_inputs, training_outputs = self.get_data(file_train, 500, device)
        testing_inputs, testing_outputs = self.get_data(file_test, 500, device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def get_data(self, file, n_samples, device):
        
        if n_samples==1:
            n_samples=500
            one = True
        else:
            one = False  
        
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, 8, device)
        
        input_data = torch.tensor(np.zeros((n_samples, 1 + 2*self.N_Fourier_F, 64, 64))).type(torch.float32).to(device)
        output_data = torch.tensor(np.zeros((n_samples, 1, 64, 64))).type(torch.float32).to(device)
        with h5py.File(file, "r") as f:
            grid = torch.tensor(f["grid"][:]).type(torch.float32).to(device)
            
            if self.N_Fourier_F>0:
                ff_grid = FF(grid).permute(2,0,1)
            
            
            for i in range(n_samples):
                if one:
                    name = "sample_" + str(0)
                else:
                    name = "sample_" + str(i)
                input_dist = torch.tensor(f[name]["initial_dist"][:].reshape(64, 64)).type(torch.float32).to(device)
                output_dist = torch.tensor(f[name]["final_dist"][:].reshape(64, 64)).type(torch.float32).to(device)
                # p = f[name]["initial_pos"][:]

                # pos = np.ones((2,64,64))
                # pos[0] = p[0]*pos[0]
                # pos[1] = p[1]*pos[1]

                # input_data[i] =np.concatenate((grid,input.reshape(1, 32, 32)),0)
                input_data[i, 0] = input_dist
                
                if self.N_Fourier_F>0:
                    input_data[i,1:] = ff_grid
                # input_data[i][1:] = pos

                output_data[i, 0] = output_dist

                #print(name)
                #print(input_dist[0,0,0])
       

        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        with h5py.File("data/RotationTranslation/TrainingData_RotationTranslation.h5", "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32).permute(2, 0, 1)).numpy()
        return grid

#------------------------------------------------------------------------------------------------------------------------------------

class Rotation:
    def __init__(self, network_properties, device, batch_size):
        
        
        #inputs, outputs = self.get_data(2000)
        file_train = "data/RotationTranslation/TrainingData_Rotation.h5"
        file_test = "data/RotationTranslation/TestData_Rotation.h5"
        
        training_inputs, training_outputs = self.get_data(file_train, 1)
        testing_inputs, testing_outputs = self.get_data(file_test, 500)

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        
        torch.manual_seed(retrain)

        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def get_data(self, file, n_samples):
        
        if n_samples==1:
            n_samples=500
            one = True
        else:
            one = False  
        
        input_data = np.zeros((n_samples, 1, 64, 64))
        output_data = np.zeros((n_samples, 1, 64, 64))
        with h5py.File(file, "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32).permute(2, 0, 1)).numpy()
            print("GRID ", grid.shape)
            for i in range(n_samples):
                if one:
                    name = "sample_" + str(0)
                else:
                    name = "sample_" + str(i)
                input_dist = f[name]["initial_dist"][:].reshape(1, 64, 64)
                output_dist = f[name]["final_dist"][:].reshape(1, 64, 64)
                # p = f[name]["initial_pos"][:]

                # pos = np.ones((2,64,64))
                # pos[0] = p[0]*pos[0]
                # pos[1] = p[1]*pos[1]

                # input_data[i] =np.concatenate((grid,input.reshape(1, 32, 32)),0)
                input_data[i] = input_dist
                # input_data[i][1:] = pos

                output_data[i] = output_dist

                #print(name)
                #print(input_dist[0,0,0])
       

        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        with h5py.File("data/RotationTranslation/TrainingData_Rotation.h5", "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32).permute(2, 0, 1)).numpy()
        return grid

#------------------------------------------------------------------------------------------------------------------------------------

class TranslationRotation:
    def __init__(self, network_properties, device, batch_size):
        # inputs, outputs = self.get_data(2000)
        file_train = "data/RotationTranslation/TrainingData_RotationEasier.h5"
        file_test = "data/RotationTranslation/TestData_RotationEasier.h5"

        training_inputs, training_outputs = self.get_data(file_train, 500)
        testing_inputs, testing_outputs = self.get_data(file_test, 500)

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)

        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def get_data(self, file, n_samples):
        input_data = np.zeros((n_samples, 1, 64, 64))
        output_data = np.zeros((n_samples, 1, 64, 64))
        with h5py.File(file, "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32).permute(2, 0, 1)).numpy()
            for i in range(n_samples):
                name = "sample_" + str(i)
                #velocity = f[name]["velocity"][:]
                #velocity_tens_x = np.full((1, 64, 64), velocity[0])
                #velocity_tens_y = np.full((1, 64, 64), velocity[1])
                input_dist = f[name]["initial_dist"][:].reshape(1, 64, 64)
                output_dist = f[name]["final_dist"][:].reshape(1, 64, 64)

                #print(input_dist)

                #input_dist = np.concatenate((velocity_tens_x, velocity_tens_y, input_dist), 0)
                # p = f[name]["initial_pos"][:]

                # pos = np.ones((2,64,64))
                # pos[0] = p[0]*pos[0]
                # pos[1] = p[1]*pos[1]

                # input_data[i] =np.concatenate((grid,input.reshape(1, 32, 32)),0)
                input_data[i] = input_dist
                # input_data[i][1:] = pos

                output_data[i] = output_dist

            return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        with h5py.File("data/RotationTranslation/TestData_RotationEasier.h5", "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32).permute(2, 0, 1)).numpy()
        return grid




#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

#Useless for now

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------


class NavierStokes:
    def __init__(self, network_properties, device, batch_size):
        training_inputs, training_outputs = self.get_data("data/data_32_32__Navier_Stokes/training_data.hdf5", 500)
        testing_inputs, testing_outputs = self.get_data("data/data_32_32__Navier_Stokes/validation_data.hdf5", 100)

        training_inputs, testing_inputs = self.normalize(training_inputs, testing_inputs)
        training_outputs, testing_outputs = self.normalize(training_outputs, testing_outputs)

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        
        torch.manual_seed(retrain)

        self.model = ContConv2D(in_channels=1 + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data, data_t):
        M = max(torch.max(data).item(), torch.max(data_t).item())
        m = min(torch.min(data).item(), torch.min(data_t).item())

        return (data - m) / (M - m), (data_t - m) / (M - m)

    def get_data(self, file, n_samples):
        input_data = np.zeros((n_samples, 1, 32, 32))
        output_data = np.zeros((n_samples, 1, 32, 32))
        with h5py.File(file, "r") as f:
            input_data = np.zeros((n_samples, 1, 32, 32))
            output_data = np.zeros((n_samples, 1, 32, 32))
            a_group_key = list(f.keys())
            for i in range(n_samples):
                s_in = "input_" + str(i)
                s_out = "output_" + str(i)

                input_data[i] = torch.tensor(np.array(f[a_group_key[0]][s_in])).reshape(1, 32, 32)
                output_data[i] = torch.tensor(np.array(f[a_group_key[1]][s_out]).reshape(-1, 32, 32)[-1]).reshape(1, 32, 32)

            return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):
        with h5py.File("data/data_32_32__Navier_Stokes/training_data.hdf5", "r") as f:
            a_group_key = list(f.keys())

            input_coords = f[a_group_key[0]]["input_coords"]
            index = torch.tensor(input_coords).reshape(32, 32)

            grid = torch.zeros((2, 32, 32))

            for i in range(32):
                for j in range(32):
                    grid[0][i, j] = (float(index[i, j]) // 32) / 31.0
                    grid[1][i, j] = (float(index[i, j]) % 32) / 31.0
        return grid

class NEMO:
    def __init__(self, network_properties, device, batch_size, cluster = True, notebook = False):

        # training_outputs = torch.nn.functional.interpolate(training_outputs, (32,32), mode = "bicubic")
        # testing_outputs = torch.nn.functional.interpolate(testing_outputs, (32,32), mode = "bicubic")

        in_size = network_properties["in_size"]
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        self.N_Fourier_F = network_properties["FourierF"]
        self.in_size = 100
        self.arg__size = 3

        train_a, train_u, test_a, test_u = self.get_data(['ssh', 'bath', 'mslp'], 12)
        
        
        #train_a = train_a.permute(0, 3, 1, 2)
        #train_u = train_u.permute(0, 3, 1, 2)
        #test_a = test_a.permute(0, 3, 1, 2)
        #test_u = test_u.permute(0, 3, 1, 2)

        print(torch.max(train_a))
        
        torch.manual_seed(retrain)
        

        self.model = ContConv2D(in_channels = self.arg__size + 2*self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial = radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size
                                ).to(device)

        self.train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    # def normalize(self, data, data_t):
    #    M = max(torch.max(data).item(), torch.max(data_t).item())
    #    m = min(torch.min(data).item(), torch.min(data_t).item())
    #    
    #    return (data - m)/(M-m), (data_t - m)/(M-m)

    def get_data(self, arguments, n_months):
        
        file_ssh = 'data/ssh_training_data.h5'
        file_mslp = 'data/mslp_training_data.h5'
        file_ssh_test = 'data/ssh_test_data.h5'
        file_mslp_test = 'data/mslp_test_data.h5'

        file_bath = "data/bathymetry.h5"
        
        
        file_ssh = '/cluster/scratch/braonic/nemo/ssh_training_data.h5'
        file_mslp = '/cluster/scratch/braonic/nemo/mslp_training_data.h5'
        file_ssh_test = '/cluster/scratch/braonic/nemo/ssh_test_data.h5'
        file_mslp_test = '/cluster/scratch/braonic/nemo/mslp_test_data.h5'

        file_bath = "/cluster/scratch/braonic/nemo/bathymetry.h5"
        
        
        L0 = 260
        R0 = 360
        L1 = 192
        R1 = 292
        dL = 6
        Len1 = 1
        Len2 = 1

        for i in range(len(arguments)):

            v = arguments[i]

            if v != 'bath':

                # load inputs & outputs (train & test) ==> N = ntrain + ntest
                if v == 'ssh':
                    inputs_v, outputs_v = load_input_output(file=file_ssh, n_months=n_months, dL=dL, Len1=Len1, Len2=Len2, size_x=100, size_y=100)
                    inputs_test_v, outputs_test_v = load_input_output(file=file_ssh_test, n_months=n_months, dL=dL, Len1=Len1, Len2=Len2, size_x=100, size_y=100)

                elif v == 'mslp':
                    inputs_v, outputs_v = load_input_output(file=file_mslp, n_months=n_months, dL=dL, Len1=Len1, Len2=Len2, size_x=100, size_y=100)
                    inputs_test_v, outputs_test_v = load_input_output(file=file_mslp_test, n_months=n_months, dL=dL, Len1=Len1, Len2=Len2, size_x=100, size_y=100)

                # cat in order to normalize them
                inputs_V = torch.cat((inputs_v, inputs_test_v), dim=0)
                outputs_V = torch.cat((outputs_v, outputs_test_v), dim=0)

                ntrain = inputs_v.shape[0]
                N = inputs_V.shape[0]

                # normalize (min/max normalizer)
                inputs_V, outputs_V, extrema = normalize_encode(inputs_V, outputs_V)

                if i == 0:
                    outputs = outputs_V

                    inputs = inputs_V
                else:

                    inputs = torch.cat((inputs, inputs_V), 3)

            elif v == 'bath':
                bath = load_bathymetry(file_bath, L0, R0, L1, R1)
                bath, extrema_bath = normalize_bath_encode(bath)
                alpha_bath = 0.05

                # bath = bath.reshape(1,1,100,100)
                # bath = torch.nn.functional.interpolate(bath, (64,64), mode = "bicubic")
                # bath = bath_transform(bath, alpha_bath, extrema_bath)

                bath = bath.repeat(N, 1, 1)
                bath = bath.reshape(bath.shape[0], bath.shape[1], bath.shape[2], 1)

                inputs = torch.cat((inputs, bath), 3)
        
        n_samples = inputs.shape[0]
        input_data = np.zeros((n_samples, self.arg__size + 2*self.N_Fourier_F, self.in_size, self.in_size))
        output_data = np.zeros((n_samples, 1, self.in_size, self.in_size))
        
        grid = self.get_grid()
        if self.N_Fourier_F>0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
        
        if self.N_Fourier_F>0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2,0,1)
            ff_grid = ff_grid.detach().cpu().numpy()
        
        print(inputs.shape)
        print(inputs)
        inputs = torch.permute(inputs, (0, 3, 1, 2))
        #inputs = torch.nn.functional.interpolate(inputs, (64, 64), mode="bicubic")
        #inputs = torch.permute(inputs, (0, 2, 3, 1))
        print(inputs.shape)
        outputs = torch.permute(outputs, (0, 3, 1, 2))
        #outputs = torch.nn.functional.interpolate(outputs, (64, 64), mode="bicubic")
        #outputs = torch.permute(outputs, (0, 2, 3, 1))
        
        input_data[:,:self.arg__size,:,:] = inputs
        output_data[:,:self.arg__size,:,:] = outputs
        
        for i in range(n_samples):
            input_data[i,self.arg__size:] = ff_grid

        print(inputs.shape)
        print("AA",input_data.shape)
        print("AA",output_data.shape)
        
        return torch.tensor(input_data[:ntrain]).type(torch.float32), torch.tensor(output_data[:ntrain]).type(torch.float32), torch.tensor(input_data[ntrain:]).type(torch.float32), torch.tensor(output_data[ntrain:]).type(torch.float32)
        
    def get_grid(self):

        grid = torch.zeros((self.in_size, self.in_size,2))

        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i/(self.in_size - 1)
                grid[i, j][1] = j/(self.in_size - 1)
                
        return grid

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


class AdvectionDataset(Dataset):
    def __init__(self, which="training", nf=0):
        if which == "training":
            self.length = 1024
            self.start = 0
        elif which == "testing":
            self.length = 1024
            self.start = 1024
        self.file_data = "/cluster/scratch/braonic/data/AdvectionData.h5"
        self.file_data = "data/AdvectionData.h5"
        
        self.reader = h5py.File(self.file_data, 'r')
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 256, 256)[:,::2,::2]
        labels = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 256, 256)[:,::2,::2]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        
        file = "/cluster/scratch/braonic/data/AdvectionData.h5"
        file = "data/AdvectionData.h5"
        
        with h5py.File(file, "r") as f:
            grid = (torch.tensor(f["grid"][:]).type(torch.float32))[::2,::2]
        return grid

class Advection:
    def __init__(self, network_properties, device, batch_size):
        # inputs, outputs = self.get_data(2000)
        in_size = 128
        cutoff_den = network_properties["cutoff_den"]
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        filter_size = network_properties["filter_size"]
        kernel_size = network_properties["kernel_size"]
        retrain = network_properties["retrain"]
        radial = network_properties["radial_filter"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        self.N_Fourier_F = network_properties["FourierF"]
        torch.manual_seed(retrain)
        
        if "channel_multiplier" in network_properties:
            channel_multiplier = network_properties["channel_multiplier"]
        else:
            channel_multiplier = 32
        
        if "half_width_mult" in network_properties:
            half_width_mult = network_properties["half_width_mult"]
        else:
            half_width_mult = 1

        if "scale_factor" in network_properties:
            scale_factor = network_properties["scale_factor"]
        else:
            scale_factor = 2

        self.model = UContConv2D(in_channels=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=in_size,  # Input spatial size: int or [width, height].
                                cutoff_den=cutoff_den,
                                N_layers=N_layers,
                                N_res=N_res,
                                radial=radial,
                                filter_size=filter_size,
                                conv_kernel=kernel_size,
                                lrelu_upsampling = lrelu_upsampling,
                                half_width_mult = half_width_mult,
                                channel_multiplier = channel_multiplier,
                                scale_factor = scale_factor
                                ).to(device)

        self.train_loader = DataLoader(AdvectionDataset("training", self.N_Fourier_F), batch_size=batch_size, shuffle=True, num_workers=16)
        self.test_loader = DataLoader(AdvectionDataset("testing", self.N_Fourier_F), batch_size=batch_size, shuffle=False, num_workers=16)

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------



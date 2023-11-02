import random

import h5py
import netCDF4
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from CNNModules import ConvBranch2D
from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class NavierStokes_VIDON:
    def __init__(self, network_properties, device, batch_size, training_samples=500):

        self.in_size = 33
        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        retrain = network_properties["retrain"]
        basis = network_properties["basis"]
        trunk_layers = network_properties["trunk_layers"]
        trunk_neurons = network_properties["trunk_neurons"]
        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0
        if "channel_multiply" in network_properties:
            multiply = network_properties["channel_multiply"]
        else:
            multiply = 32
        if "kernel_size" in network_properties:
            kernel_size = network_properties["kernel_size"]
        else:
            kernel_size = 3

        torch.manual_seed(retrain)

        training_inputs, training_outputs = self.get_data("data/data_32_32__Navier_Stokes/training_data_grid_time_5.hdf5", 1000)

        testing_inputs = training_inputs[500:]
        testing_outputs = training_outputs[500:]

        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]

        training_inputs, testing_inputs = self.normalize(training_inputs, testing_inputs)
        training_outputs, testing_outputs = self.normalize(training_outputs, testing_outputs)

        branch_training_inputs = training_inputs[:, 0].unsqueeze(1)
        branch_testing_inputs = testing_inputs[:, 0].unsqueeze(1)

        self.trunk_inputs = training_inputs[0, 1:].reshape(2 * self.N_Fourier_F, -1).permute(1, 0).to(device)
        # testing_inputs, testing_outputs = self.get_data("data/data_32_32__Navier_Stokes/validation_data_grid_time_5.hdf5", 32)

        branch = ConvBranch2D(in_channels=1,  # Number of input channels.
                              N_layers=N_layers,
                              N_res=N_res,
                              kernel_size=kernel_size,
                              multiply=multiply,
                              out_channel=basis).to(device)

        trunk = FeedForwardNN(2 * self.N_Fourier_F, basis, layers=trunk_layers, neurons=trunk_neurons, retrain=retrain).to(device)

        self.model = DeepOnetNoBiasOrg(branch, trunk).to(device)

        self.train_loader = DataLoader(TensorDataset(branch_training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(branch_testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data, data_t):
        M = max(torch.max(data).item(), torch.max(data_t).item())
        m = min(torch.min(data).item(), torch.min(data_t).item())

        return (data - m) / (M - m), (data_t - m) / (M - m)

    def get_data(self, file, n_samples):
        input_data = np.zeros((n_samples, 33, 33, 1 + 2 * self.N_Fourier_F))
        output_data = np.zeros((n_samples, 33, 33, 1))

        grid = self.get_grid()

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)

        if self.N_Fourier_F > 0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.detach().cpu().numpy()

        with h5py.File(file, "r") as f:
            a_group_key = list(f.keys())

            for i in range(n_samples):
                s_in = "input_" + str(i)
                s_out = "output_" + str(i)

                input_data[i, :, :, 0] = torch.tensor(np.array(f[a_group_key[0]][s_in])).reshape(33, 33)

                if self.N_Fourier_F > 0:
                    input_data[i, :, :, 1:] = ff_grid

                output_data[i, :, :, 0] = torch.tensor(np.array(f[a_group_key[1]][s_out])).reshape(33, 33)

            return torch.tensor(input_data).type(torch.float32).permute(0, 3, 1, 2), torch.tensor(output_data).type(torch.float32).permute(0, 3, 1, 2)

    def get_grid(self):
        grid = torch.zeros((33, 33, 2))

        for i in range(33):
            for j in range(33):
                grid[i, j][0] = i / 32.0
                grid[i, j][1] = j / 32.0
        return grid


# ------------------------------------------------------------------------------------------------------------------------------------
import scipy


def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real


def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1 / N_old)
    sel = np.logical_and(freqs >= -N / 2, freqs <= N / 2 - 1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:, :, sel, :][:, :, :, sel]
    u_down = samples_ifft(u_hat_down)
    return u_down


# ------------------------------------------------------------------------------------------------------------------------------------

from CNO_Processing.FourierFeatures import FourierFeatures


class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, in_size=64, file="ddsl_N128/", cluster=False):

        self.in_size = in_size

        assert self.in_size <= 128

        N_layers = network_properties["N_layers"]
        N_res = network_properties["N_res"]
        retrain = network_properties["retrain"]
        basis = network_properties["basis"]
        trunk_layers = network_properties["trunk_layers"]
        trunk_neurons = network_properties["trunk_neurons"]
        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0
        if "channel_multiply" in network_properties:
            multiply = network_properties["channel_multiply"]
        else:
            multiply = 32
        if "kernel_size" in network_properties:
            kernel_size = network_properties["kernel_size"]
        else:
            kernel_size = 3

        torch.manual_seed(retrain)

        training_inputs, training_outputs = self.get_data("data/Euler_Shear/" + file, 1024, cluster)

        training_inputs = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)

        testing_inputs = training_inputs[896:]
        testing_outputs = training_outputs[896:]
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]

        branch_training_inputs = training_inputs[:, 0].unsqueeze(1)
        branch_testing_inputs = testing_inputs[:, 0].unsqueeze(1)

        self.trunk_inputs = training_inputs[0, 1:].reshape(2 * self.N_Fourier_F, -1).permute(1, 0).to(device)
        # testing_inputs, testing_outputs = self.get_data("data/data_32_32__Navier_Stokes/validation_data_grid_time_5.hdf5", 32)

        branch = ConvBranch2D(in_channels=1,  # Number of input channels.
                              N_layers=N_layers,
                              N_res=N_res,
                              kernel_size=kernel_size,
                              multiply=multiply,
                              out_channel=basis).to(device)

        trunk = FeedForwardNN(2 * self.N_Fourier_F, basis, layers=trunk_layers, neurons=trunk_neurons, retrain=retrain).to(device)

        self.model = DeepOnetNoBiasOrg(branch, trunk).to(device)

        self.train_loader = DataLoader(TensorDataset(branch_training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(branch_testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data):
        m = torch.max(data)
        M = torch.min(data)
        return (data - m) / (M - m)

    def get_data(self, folder, n_samples, cluster):

        input_data = np.zeros((n_samples, 1 + 2 * self.N_Fourier_F, self.in_size, self.in_size))
        output_data = np.zeros((n_samples, 1, self.in_size, self.in_size))

        grid = self.get_grid()
        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)

        if self.N_Fourier_F > 0:
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            ff_grid = ff_grid.detach().cpu().numpy()

        for i in range(n_samples):

            if i >= 896:
                if cluster:
                    folder = "/cluster/scratch/braonic/Euler_Shear/ddsl_N128/"
                else:
                    folder = "data/Euler_Shear/ddsl_N128/"

            file_input = folder + "sample_" + str(i) + "_time_0.nc"
            file_output = folder + "sample_" + str(i) + "_time_10.nc"

            f = netCDF4.Dataset(file_input, 'r')
            if self.in_size < 128:
                input_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1, 1, 128, 128), self.in_size)
            else:
                input_data[i, 0] = np.array(f.variables['u'][:])
            f.close()

            f = netCDF4.Dataset(file_output, 'r')
            if self.in_size < 128:
                output_data[i, 0] = downsample(np.array(f.variables['u'][:]).reshape(1, 1, 128, 128), self.in_size)
            else:
                output_data[i, 0] = np.array(f.variables['u'][:])
            f.close()

            if self.N_Fourier_F > 0:
                input_data[i, 1:, :, :] = ff_grid

        return torch.tensor(input_data).type(torch.float32), torch.tensor(output_data).type(torch.float32)

    def get_grid(self):

        grid = torch.zeros((self.in_size, self.in_size, 2))

        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i / (self.in_size - 1)
                grid[i, j][1] = j / (self.in_size - 1)

        return grid

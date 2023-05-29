import random

import h5py
import netCDF4
import numpy as np
# ------------------------------------------------------------------------------------------------------------------------------------
import scipy
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

from BaselinesModules import UNetOrg
from CNNModules import ConvBranch2D
from training.FourierFeatures import FourierFeatures
from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg


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


class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, file="ddsl_N128/", cluster=False, insample=True):

        self.in_size = 64

        assert self.in_size <= 128

        retrain = network_properties["retrain"]
        channels = network_properties["channels"]
        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0

        torch.manual_seed(retrain)

        self.model = UNetOrg(1 + 2 * self.N_Fourier_F, 1, channels).to(device)

        if insample:
            file = "data/Euler_Shear_0_05/"
            folder = "data/Euler_Shear/"

            training_inputs, training_outputs = self.get_data(folder + file, 1024, cluster)

            training_inputs = self.normalize(training_inputs, inp=True)
            training_outputs = self.normalize(training_outputs, inp=False)

            validation_inputs = training_inputs[896:]
            validation_outputs = training_outputs[896:]
            testing_inputs = training_inputs[750:878]
            testing_outputs = training_outputs[750:878]
            training_inputs = training_inputs[:training_samples]
            training_outputs = training_outputs[:training_samples]

            self.train_loader = DataLoader(TensorDataset(training_inputs, training_outputs), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(TensorDataset(validation_inputs, validation_outputs), batch_size=batch_size, shuffle=False)
            self.test_loader_2 = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

        else:
            file = "ddsl_N128_out/"
            folder = "data/Euler_Shear/"

            testing_inputs, testing_outputs = self.get_data(folder + file, 1024, cluster)
            m = 1.4294605255126953
            M = -1.4294605255126953
            testing_inputs = self.normalize_out(testing_inputs, m, M)
            m = 2.0602376461029053
            M = -2.0383081436157227
            testing_outputs = self.normalize_out(testing_outputs, m, M)

            testing_inputs = testing_inputs[750:878]
            testing_outputs = testing_outputs[750:878]

            self.test_loader_2 = DataLoader(TensorDataset(testing_inputs, testing_outputs), batch_size=batch_size, shuffle=False)

    def normalize(self, data, inp=True):
        m = torch.max(data)
        M = torch.min(data)

        return (data - m) / (M - m)

    def normalize_out(self, data, m, M, inp=True):
        # m = torch.max(data)
        # M = torch.min(data)

        # print(m.item(), M.item())
        return (data - m) / (M - m)

    def get_data(self, folder, n_samples, insample):

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


class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=1024, insample=True):

        self.reader = h5py.File("data/PoissonData_IN_TRAINING.h5", 'r')

        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]

        if which == "training":
            self.length = training_samples
            self.start = 0
            self.file_data = "data/PoissonData_IN_TRAINING.h5"
        elif which == "validation":
            self.length = 128
            self.start = training_samples
            self.file_data = "data/PoissonData_IN_TRAINING.h5"

        elif which == "testing":

            if insample:
                self.length = 256
                self.start = 1024 + 128
                self.file_data = "data/PoissonData_IN_TRAINING.h5"
            else:
                self.length = 256
                self.start = 0
                #self.file_data = "data/PoissonData_OUT1_decay04.h5"
                self.file_data = "data/PoissonData_OUT2_20modes.h5"
        self.reader = h5py.File(self.file_data, 'r')

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

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


class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=512, t=5, insample=True):
        self.reader = h5py.File("data//WaveData_IN_24modes.h5", 'r')

        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        if which == "training":
            self.length = training_samples
            self.start = 0
            self.file_data = "data/WaveData_IN_24modes.h5"
        elif which == "validation":
            self.length = 128
            self.start = training_samples
            self.file_data = "data/WaveData_IN_24modes.h5"

        elif which == "testing":
            if insample:
                self.length = 256
                self.start = 1024 + 128
                self.file_data = "data/WaveData_IN_24modes.h5"
            else:
                self.length = 256
                self.start = 0
                self.file_data = "data/WaveData_OUT_32modes_085decay.h5"

        self.in_size = 64

        self.reader = h5py.File(self.file_data, 'r')

        self.t = t
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start) + "_t_" + str(self.t)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start) + "_t_" + str(self.t)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs, labels

    def get_grid(self):
        grid = torch.zeros((self.in_size, self.in_size, 2))

        for i in range(self.in_size):
            for j in range(self.in_size):
                grid[i, j][0] = i / (self.in_size - 1)
                grid[i, j][1] = j / (self.in_size - 1)

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid


class AllenCahnDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=256, insample=True):
        self.reader = h5py.File("data/AllenCahn_NEW.h5", 'r')

        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        if which == "training":
            self.length = training_samples
            self.start = 0
            self.file_data = "data/AllenCahn_NEW.h5"

        elif which == "validation":
            self.length = 128
            self.start = training_samples
            self.file_data = "data/AllenCahn_NEW.h5"

        elif which == "testing":
            if insample:
                self.length = 128
                self.start = 256 + 128
                self.file_data = "data/AllenCahn_NEW.h5"
            else:
                self.length = 128
                self.start = 0
                self.file_data = "data/AllenCahn_OUT_16modes_random_decay_085_115.h5"
        self.reader = h5py.File(self.file_data, 'r')

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

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


class ContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=512, insample=True):
        if which == "training":
            self.length = training_samples
            self.start = 0

            self.file_data = "data/ContTranslation.h5"

        elif which == "validation":
            self.length = 256
            self.start = 0

            self.file_data = "data/ContTranslation_test_in_sample.h5"

        elif which == "testing":

            if insample:
                self.length = 256
                self.start = 256
                self.file_data = "data/ContTranslation_test_in_sample.h5"
            else:
                self.length = 256
                self.start = 256 + 512
                self.file_data = "data/ContTranslation.h5"

        self.reader = h5py.File(self.file_data, 'r')

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

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


class DiscContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=512, insample=True):
        if which == "training":
            self.length = training_samples
            self.start = 0
            self.file_data = "data/DiscTranslation.h5"

        elif which == "validation":
            self.length = 256
            self.start = 512
            self.file_data = "data/DiscTranslation.h5"

        elif which == "testing":
            if insample:
                self.length = 256
                self.start = 512 + 256
                self.file_data = "data/DiscTranslation.h5"
            else:
                self.length = 256
                self.start = 1024
                self.file_data = "data/DiscTranslation.h5"

        self.reader = h5py.File(self.file_data, 'r')

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

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


class AirfoilDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=512, insample=True):
        if which == "training":
            self.length = training_samples
            self.start = 0

        elif which == "validation":
            self.length = 128
            self.start = 750

        elif which == "testing":
            self.length = 128
            self.start = 750 + 128

        if insample:
            self.file_data = "data/Airfoil.h5"
        else:
            if which == "testing":
                self.length = 128
                self.start = 0
            self.file_data = "data/Airfoil_out.h5"

        self.reader = h5py.File(self.file_data, 'r')

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32)[90:90 + 256, 128:128 + 256][::2, ::2].reshape(1, 128, 128)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32)[90:90 + 256, 128:128 + 256][::2, ::2].reshape(1, 128, 128)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 128)
        y = torch.linspace(0, 1, 128)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid


class EquationModel:
    def __init__(self, network_properties, device, batch_size, which_data, training_samples=512, insample=True):
        # inputs, outputs = self.get_data(2000)

        retrain = network_properties["retrain"]
        channels = network_properties["channels"]
        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0

        torch.manual_seed(retrain)

        self.model = UNetOrg(1 + 2 * self.N_Fourier_F, 1, channels).to(device)

        if which_data == "poisson":
            training_set = SinFrequencyDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = SinFrequencyDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = SinFrequencyDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)
        elif which_data == "wave":
            training_set = WaveEquationDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = WaveEquationDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = WaveEquationDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)
        elif which_data == "allen_cahn":
            training_set = AllenCahnDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = AllenCahnDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = AllenCahnDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)
        elif which_data == "cont_t":
            training_set = ContTranslationDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = ContTranslationDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = ContTranslationDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)
        elif which_data == "discont_t":
            training_set = DiscContTranslationDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = DiscContTranslationDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = DiscContTranslationDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)

        elif which_data == "airfoil":
            training_set = AirfoilDataset("training", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            validation_set = AirfoilDataset("validation", self.N_Fourier_F, training_samples=training_samples, insample=insample)
            testing_set = AirfoilDataset("testing", self.N_Fourier_F, training_samples=training_samples, insample=insample)
        else:
            raise ValueError("not defined which_data")

        self.train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader_2 = DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=0)

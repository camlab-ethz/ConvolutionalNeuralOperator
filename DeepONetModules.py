import numpy as np
import torch.nn as nn

from debug_tools import *


def kaiming_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0.01, nonlinearity='leaky_relu')
        torch.nn.init.zeros_(m.bias.data)




class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['mish']:
        return nn.Mish()
    elif name in ['sin']:
        return Sin()
    else:
        raise ValueError('Unknown activation function')


def init_xavier(model):
    torch.manual_seed(model.retrain)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if model.act_string == "tanh" or model.act_string == "relu" or model.act_string == "leaky_relu":
                gain = nn.init.calculate_gain(model.act_string)
            else:
                gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)

    model.apply(init_weights)


class FeedForwardNN(nn.Module):

    def __init__(self, input_dimension, output_dimension, layers=8, neurons=256, retrain=4):
        super(FeedForwardNN, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = layers
        self.neurons = neurons
        self.act_string = "leaky_relu"
        self.retrain = retrain

        torch.manual_seed(self.retrain)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.batch_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.activation = activation(self.act_string)

        self.apply(kaiming_init)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, (l, b) in enumerate(zip(self.hidden_layers, self.batch_layers)):
            x = b(self.activation(l(x)))
        return self.output_layer(x)

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class DeepOnetNoBiasOrg(nn.Module):
    def __init__(self, branch, trunk):
        super(DeepOnetNoBiasOrg, self).__init__()
        self.branch = branch
        self.trunk = trunk
        self.b0 = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.p = self.trunk.output_dimension

    def forward(self, u_, x_):
        nx = int(x_.shape[0]**0.5)
        weights = self.branch(u_)
        basis = self.trunk(x_)
        out = (torch.matmul(weights, basis.T) + self.b0) / self.p ** 0.5
        out = out.reshape(-1, 1, nx, nx)

        return out


    def get_n_params(self):
        pp = 0

        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams

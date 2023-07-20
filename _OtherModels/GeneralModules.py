import torch
import torch.nn as nn


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
    elif name in ['softsign', 'Softsign', "SoftSign"]:
        return nn.Softsign()
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

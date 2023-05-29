import torch.nn as nn
import torch.nn.functional as F

from GeneralModules import activation, init_xavier
from debug_tools import *


class Resnet(nn.Module):
    """
    Implements the physics-informed neural network.
    """

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, retrain):
        super(Resnet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = n_hidden_layers
        self.neurons = neurons
        self.act_string = "leaky_relu"
        self.retrain = retrain
        self.p = 0
        self.activation = activation(self.act_string)
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        n_res_blocks = (self.n_hidden_layers - 1) // 2
        n_remaining_layers = (self.n_hidden_layers - 1) % 2
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.neurons, self.act_string, self.retrain) for _ in range(n_res_blocks)])
        if n_remaining_layers == 0:
            self.remaining_layers = None
            self.remaining_batch_layers = None
        else:
            self.remaining_layers = nn.ModuleList(
                [nn.Linear(self.neurons, self.neurons) for _ in range(n_remaining_layers)])
            self.remaining_batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.neurons) for _ in range(n_remaining_layers)])

        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.dropout = nn.Dropout(self.p)
        init_xavier(self)

    def forward(self, x):
        batch_s = x.shape[0]
        res_x = x.shape[2]
        res_y = x.shape[3]
        c = x.shape[1]
        x = x.reshape(x.shape[0], -1)
        x = self.activation(self.input_layer(x))
        for block in self.residual_blocks:
            x = self.dropout(block(x))
        for l, b in zip(self.remaining_layers, self.remaining_batch_layers):
            x = self.dropout(b(self.activation(l(x))))

        x = self.output_layer(x)
        return x.reshape(batch_s, 1, res_x, res_y)
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class ResidualBlock(nn.Module):
    """
    Defines a residual block consisting of two layers.
    """

    def __init__(self, num_neurons, activation_f, retrain):
        super(ResidualBlock, self).__init__()
        self.neurons = num_neurons
        self.act_string = activation_f
        self.activation = activation(activation_f)

        self.layer_1 = nn.Linear(self.neurons, self.neurons)
        self.batch_1 = nn.BatchNorm1d(self.neurons)

        self.layer_2 = nn.Linear(self.neurons, self.neurons)
        self.batch_2 = nn.BatchNorm1d(self.neurons)
        self.retrain = retrain
        init_xavier(self)

    def forward(self, x):
        z = self.batch_1(self.activation(self.layer_1(x)))
        z = self.batch_2(self.layer_2(z))
        z = z + x
        return self.activation(z)


""" Parts of the U-Net model and 2D Convolution"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, start, print_bool=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        self.print = print_bool

        self.inc = DoubleConv(n_channels, start)
        self.down1 = Down(start, start * 2)
        self.down2 = Down(start * 2, start * 2)
        self.down3 = Down(start * 2, start * 4)
        self.down4 = Down(start * 4, start * 4)
        self.down5 = Down(start * 4, start * 8)
        factor = 2 if self.bilinear else 1
        self.down6 = Down(start * 8, start * 8)

        self.up1 = Up(start * 8, start * 8, self.bilinear)
        self.up2 = Up(start * 8, start * 4, self.bilinear)
        self.up3 = Up(start * 4, start * 4, self.bilinear)
        self.up4 = Up(start * 4, start * 2, self.bilinear)
        self.up5 = Up(start * 2, start * 2, self.bilinear)
        self.up6 = Up(start * 2, start, self.bilinear)
        self.outc = OutConv(start, n_classes)

    def forward(self, x):
        if self.print: print(x.shape)
        x1 = self.inc(x)
        if self.print: print(x1.shape)
        x2 = self.down1(x1)
        if self.print: print(x2.shape)
        x3 = self.down2(x2)
        if self.print: print(x3.shape)
        x4 = self.down3(x3)
        if self.print: print(x4.shape)
        x5 = self.down4(x4)
        if self.print: print(x5.shape)
        x6 = self.down5(x5)
        if self.print: print(x6.shape)
        x7 = self.down6(x6)
        if self.print: print(x7.shape)
        x = self.up1(x7, x6)
        if self.print: print(x.shape)
        x = self.up2(x, x5)
        if self.print: print(x.shape)
        x = self.up3(x, x4)
        if self.print: print(x.shape)
        x = self.up4(x, x3)
        if self.print: print(x.shape)
        x = self.up5(x, x2)
        if self.print: print(x.shape)
        x = self.up6(x, x1)
        if self.print: print(x.shape)
        logits = self.outc(x)
        if self.print: print(logits.shape)
        if self.print: quit()
        return logits

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams



class UNetOrg(nn.Module):
    def __init__(self, n_channels, n_classes, channels, bilinear=False):
        super(UNetOrg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, channels))
        self.down1 = (Down(channels, 2*channels))
        self.down2 = (Down(2*channels, 4*channels))
        self.down3 = (Down(4*channels, 8*channels))
        factor = 2 if bilinear else 1
        self.down4 = (Down(8*channels, 16*channels // factor))
        self.up1 = (Up(16*channels, 8*channels // factor, bilinear))
        self.up2 = (Up(8*channels, 4*channels // factor, bilinear))
        self.up3 = (Up(4*channels, 2*channels // factor, bilinear))
        self.up4 = (Up(2*channels, channels, bilinear))
        self.outc = (OutConv(channels, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams

import torch.nn as nn
import torch.nn.functional as F

from debug_tools import *


# ------------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 kernel_size=3):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.cont_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                   kernel_size=self.kernel_size, stride=1,
                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
        self.sigma = F.leaky_relu
        # conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size, 1, 1)
        # bn = torch.nn.InstanceNorm2d(num_features)
        # relu = torch.nn.ReLU(True)
        # relu = torch.nn.GELU()

        # self.resnet_block = torch.nn.Sequential(
        #    cont_conv,
        #    conv)

    def forward(self, x):
        # out = self.cont_conv(x)
        # out = self.sigma(out)
        p = 0.1
        return self.cont_conv(self.sigma(x)) * p + (1 - p) * x


class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 N_layers,  # Number of layers in the network
                 N_res,
                 kernel_size=3,
                 multiply=32
                 ):  # Number of ResNet Blocks

        super(Conv2D, self).__init__()

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        self.N_layers = N_layers

        #######################################################################################

        self.channel_multiplier = multiply

        in_size = 33
        self.in_size = 33

        self.feature_maps = [in_channels]
        for i in range(0, self.N_layers // 2):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        for i in range(self.N_layers // 2 - 2, -1, -1):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        self.feature_maps.append(1)

        # Define the size of the data for each layer (note that we downsample/usample)
        self.size = []
        for i in range(0, self.N_layers // 2 + 1):
            self.size.append(in_size // (2 ** i))
        for i in range(self.N_layers // 2 - 1, -1, -1):
            self.size.append(in_size // (2 ** i))

        # Define the sizes & number of channels for layers where you x2 upsample and x2 downsample 
        # Note: no changing in the sampling rate in the end of this operation
        # Note: we call this operation size_invariant
        # Note: the size and # of feature maps is the same as before, but we define them for convenience

        self.size_inv = self.size[:-1]
        self.feature_maps_invariant = self.feature_maps[:-1]

        print("size: ", self.size)
        print("channels: ", self.feature_maps)

        assert len(self.feature_maps) == self.N_layers + 1

        self.kernel_size = kernel_size
        self.cont_conv_layers = nn.ModuleList([nn.Conv2d(self.feature_maps[i],
                                                         self.feature_maps[i + 1],
                                                         kernel_size=self.kernel_size, stride=1,
                                                         padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                               for i in range(N_layers)])

        self.cont_conv_layers_invariant = nn.ModuleList([nn.Conv2d(self.feature_maps_invariant[i],
                                                                   self.feature_maps_invariant[i],
                                                                   kernel_size=self.kernel_size, stride=1,
                                                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                                         for i in range(N_layers)])

        self.sigma = F.leaky_relu
        """
        self.cont_conv_layers_invariant = nn.ModuleList([(SynthesisLayer(w_dim=1,
                                                                         is_torgb=False,
                                                                         is_critically_sampled=False,
                                                                         use_fp16=False,

                                                                         # Input & output specifications.
                                                                         in_channels=self.feature_maps_invariant[i],
                                                                         out_channels=self.feature_maps_invariant[i],
                                                                         in_size=self.size_inv[i],
                                                                         out_size=self.size_inv[i],
                                                                         in_sampling_rate=self.size_inv[i],
                                                                         out_sampling_rate=self.size_inv[i],
                                                                         in_cutoff=self.size_inv[i] // cutoff_den,
                                                                         out_cutoff=self.size_inv[i] // cutoff_den,
                                                                         in_half_width=in_half_width,
                                                                         out_half_width=out_half_width))
                                                         for i in range(N_layers)])
        """

        # Define the resnet block --> ##### TO BE DISCUSSED ######

        self.resnet_blocks = []

        # print(self.feature_maps[self.N_layers // 2], )
        for i in range(N_res):
            self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[self.N_layers // 2]))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.N_res = N_res

        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.downsample2 = nn.AvgPool2d(2, stride=2, padding=0)
        self.downsample4 = nn.AvgPool2d(4, stride=4, padding=1)

        self.last = nn.Upsample(size=in_size)

    def forward(self, x):
        # Execute the left part of the network
        # print("")
        for i in range(self.N_layers // 2):
            # print("BEFORE I1",x.shape)
            y = self.cont_conv_layers_invariant[i](x)
            # print("After I1",y.shape)
            y = self.sigma(self.upsample2(y))
            # print("After US1",y.shape)
            x = self.downsample2(y)
            # print("AFTER IS1",x.shape)

            # print("INV DONE")
            y = (self.cont_conv_layers[i](x))
            # print("AFTER CONTCONV", y.shape)
            y = self.upsample2(y)
            # print("AFTER UP", y.shape)
            x = self.downsample4(self.sigma(y))
            # print("AFTER IS2",x.shape)

        for i in range(self.N_res):
            x = self.resnet_blocks[i](x)
            # print("RES",x.shape)

        # Execute the right part of the network
        for i in range(self.N_layers // 2, self.N_layers):
            x = self.downsample2(self.sigma(self.upsample2(self.cont_conv_layers_invariant[i](x))))
            # print("AFTER INV",x.shape)

            x = self.downsample2(self.sigma(self.upsample4(self.cont_conv_layers[i](x))))
            # print("AFTER CONTC",x.shape)

        # print(x.shape[2])
        # print("BEFORE LAST",x.shape)
        # print("-------------")
        # print(" ")

        return self.last(x)

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


class ConvBranch2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 N_layers,  # Number of layers in the network
                 N_res,
                 out_channel=1,
                 kernel_size=3,
                 multiply=32,
                 print_bool=False
                 ):  # Number of ResNet Blocks

        super(ConvBranch2D, self).__init__()

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        self.N_layers = N_layers
        self.print_bool = print_bool
        self.channel_multiplier = multiply
        self.feature_maps = [in_channels]
        for i in range(0, self.N_layers):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        self.feature_maps_invariant = self.feature_maps

        print("channels: ", self.feature_maps)

        assert len(self.feature_maps) == self.N_layers + 1

        self.kernel_size = kernel_size
        self.cont_conv_layers = nn.ModuleList([nn.Conv2d(self.feature_maps[i],
                                                         self.feature_maps[i + 1],
                                                         kernel_size=self.kernel_size, stride=1,
                                                         padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                               for i in range(N_layers)])

        self.cont_conv_layers_invariant = nn.ModuleList([nn.Conv2d(self.feature_maps_invariant[i],
                                                                   self.feature_maps_invariant[i],
                                                                   kernel_size=self.kernel_size, stride=1,
                                                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                                         for i in range(N_layers)])

        self.sigma = F.leaky_relu

        self.resnet_blocks = []

        for i in range(N_res):
            self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[self.N_layers]))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.N_res = N_res

        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.downsample2 = nn.AvgPool2d(2, stride=2, padding=0)
        self.downsample4 = nn.AvgPool2d(4, stride=4, padding=1)

        self.flatten_layer = nn.Flatten()
        self.lazy_linear = nn.LazyLinear(out_channel)

    def forward(self, x):
        for i in range(self.N_layers):
            if self.print_bool: print("BEFORE I1", x.shape)
            y = self.cont_conv_layers_invariant[i](x)
            if self.print_bool: print("After I1", y.shape)
            y = self.sigma(self.upsample2(y))
            if self.print_bool: print("After US1", y.shape)
            x = self.downsample2(y)
            if self.print_bool: print("AFTER IS1", x.shape)

            if self.print_bool: print("INV DONE")
            y = self.cont_conv_layers[i](x)
            if self.print_bool: print("AFTER CONTCONV", y.shape)
            y = self.upsample2(y)
            if self.print_bool: print("AFTER UP", y.shape)
            x = self.downsample4(self.sigma(y))
            if self.print_bool: print("AFTER IS2", x.shape)

        for i in range(self.N_res):
            x = self.resnet_blocks[i](x)
            if self.print_bool: print("RES", x.shape)

        x = self.flatten_layer(x)
        if self.print_bool: print("Flattened", x.shape)
        x = self.lazy_linear(x)
        if self.print_bool: print("Linearized", x.shape)
        if self.print_bool: quit()
        return x

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

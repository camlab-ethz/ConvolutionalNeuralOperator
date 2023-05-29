# Implementation of the filters is borrowed from paper "Alias-Free Generative Adversarial Networks (StyleGAN3)" https://nvlabs.github.io/stylegan3/
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#-------------------------------------------------------------------------------

#CNO : SynthesisLayer does the following:
    
#   1. APPLY 2D CONVOLUTION
#   2. APPLY (MODIFIED) ACTIVATION LAYER  (upsampling -> activation function -> downsampling)

#-------------------------------------------------------------------------------

import torch.nn as nn

from debug_tools import *
from training.filtered_networks import SynthesisLayer

import numpy as np

#-------------------------------------------------------------------------------

class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 in_size,
                 cutoff_den,
                 radial,
                 conv_kernel=3,
                 filter_size=6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 1,
                 length = 1):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.in_size = in_size

        #We use w_c = s/2.0001 --> NOT critically sampled
        self.citically_sampled = False
        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff = self.in_size / cutoff_den
        
        #Define the half-width of the fiters
        self.halfwidth = half_width_mult*self.in_size - self.in_size / cutoff_den
        
        #Define parameter r
        self.length =length
        
        if self.length == 1:
            self.cont_conv = SynthesisLayer(is_critically_sampled=self.citically_sampled, # Does the layer use critical sampling?
                                            in_channels=self.in_channels,                 # Number of input channels
                                            out_channels=self.in_channels,                # Number of output channels
                                            in_size=self.in_size,                         # Input smpling rate
                                            out_size=self.in_size,                        # CNO: Output sampling rate = Input sampling rate
                                            in_sampling_rate=self.in_size,                # CNO: Input sampling rate
                                            out_sampling_rate=self.in_size,               # CNO: Output sampling rate = Input sampling rate
                                            in_cutoff= self.cutoff,                       # Input cuttof w_c
                                            out_cutoff=self.cutoff,                       # CNO: Output cuttof = Input cuttof 
                                            in_half_width= self.halfwidth,                # Input half-width
                                            out_half_width= self.halfwidth,               # CNO: Output half-width = Input half-width
                                            conv_kernel=conv_kernel,                      # Kernel size
                                            filter_size=filter_size,                      # N_tap / 2
                                            lrelu_upsampling    = lrelu_upsampling,       # N_{\sigma}
                                            use_radial_filters  = radial                  # is the filter radial?
                                            )
            
        else:
            
            self.cont_conv = nn.ModuleList([SynthesisLayer(
                                        is_critically_sampled=self.citically_sampled,  
                                        in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        in_size=self.in_size,  
                                        out_size=self.in_size, 
                                        in_sampling_rate=self.in_size,  
                                        out_sampling_rate=self.in_size,  
                                        in_cutoff= self.cutoff, 
                                        out_cutoff=self.cutoff,
                                        in_half_width= self.halfwidth,
                                        out_half_width= self.halfwidth,
                                        conv_kernel=conv_kernel,
                                        filter_size=filter_size,
                                        lrelu_upsampling    = lrelu_upsampling,
                                        use_radial_filters  = radial
                                        ) for i in range(self.length)])


    def forward(self, x):
        
        # Weighted skip connection
        p = 0.9

        if self.length==1:
            out = self.cont_conv(x)
            p = 0.9
        else:
            out = x
            for i in range(self.length):
                out = self.cont_conv[i](out)

        return (1-p)*out + p*x

#------------------------------------------------------------------------------

#CNO NETWORK:
class ContConv2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 in_size,  # Input spatial size
                 cutoff_den,
                 N_layers,  # Number of layers in the network
                 N_res,
                 radial,
                 conv_kernel=3,  # Convolution kernel size
                 filter_size=6,  # Low-pass filter size relative to the lower resolution when up/downsampling
                 lrelu_upsampling = 2,
                 half_width_mult  = 1,
                 channel_multiplier = 32,
                 length_res = 1,
                 batch_ = True,
                 lift_dim = 32,
                 out_dim = 1):

        super(ContConv2D, self).__init__()

        ###################### Define the parameters & specifications #################################################

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        # Number od (D)+(U) blocks
        self.N_layers = int(N_layers)
     
        # Lifting & Output Dimension
        self.lift_dim = lift_dim          
        self.out_dim  = out_dim
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        

        # Length of the (R) block : r parameter
        self.length_res = length_res

        
        # Critically sampled? We always use NOT citically sampled
        self.citically_sampled = False
        if cutoff_den==2:
            self.citically_sampled = True
        
        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True
        
        ###################### Define evolution of the number features ################################################

        # How the input features evolve (number of features)
        self.feature_maps = [self.lift_dim]
        for i in range(0, self.N_layers // 2):
            self.feature_maps.append(2 ** i *   self.channel_multiplier)
        
        for i in range(self.N_layers // 2 - 2, -1, -1):
            if i == self.N_layers // 2 - 1:
                self.feature_maps.append(2 ** i *   self.channel_multiplier)
            else:
                self.feature_maps.append(2* 2 ** i *   self.channel_multiplier)
        self.feature_maps.append(self.lift_dim)

        assert len(self.feature_maps) == self.N_layers + 1
        
        
        # How the output features evolve (number of features)
        self.out_feature_maps = []
        for i in range(0, self.N_layers // 2):
            self.out_feature_maps.append(2 ** i *   self.channel_multiplier)
        
        for i in range(self.N_layers // 2 - 2, -1, -1):
            self.out_feature_maps.append(2 ** i *   self.channel_multiplier)
        
        self.out_feature_maps.append(self.lift_dim)

        ###################### Define evolution of sampling rates #####################################################

        self.size = []
        for i in range(0, self.N_layers // 2 + 1):
            self.size.append(in_size // (2 ** i))
        for i in range(self.N_layers // 2 - 1, -1, -1):
            self.size.append(in_size // (2 ** i))

        # How the sampling rates of the (I) blocks evolve
        self.size_inv = self.size[:-1]
        
        # How the features of the (I) blocks evolve
        self.feature_maps_invariant = self.feature_maps[:-1]


        ###################### Define evolution of filters and their properties #######################################

        self.cutoff = np.zeros(len(self.size))
        for i in range(len(self.size)):
            self.cutoff[i] = self.size[i]/cutoff_den
        
        self.halfwidth = np.zeros(len(self.size))
        for i in range(len(self.size)):
            self.halfwidth[i] = half_width_mult*self.size[i] - self.size[i]/cutoff_den
        
        
        ###################### Define Projection & Lift ##############################################################
    
        self.project =   (SynthesisLayer(is_critically_sampled=self.citically_sampled,
                                         # Input & output specifications.
                                         in_channels =2*self.out_feature_maps[-1],
                                         out_channels = self.out_dim,
                                         in_size = self.size[-1],
                                         out_size= self.size[-1],
                                         in_sampling_rate=self.size[-1],
                                         out_sampling_rate=self.size[-1],
                                         in_cutoff = self.cutoff[-1],
                                         out_cutoff= self.cutoff[-1],
                                         in_half_width= self.halfwidth[-1],
                                         out_half_width=self.halfwidth[-1],
                                         conv_kernel=conv_kernel,  
                                         filter_size=filter_size,  
                                         lrelu_upsampling    = lrelu_upsampling,
                                         use_radial_filters  =  self.radial
                                         ))
        
        self.lift =   (SynthesisLayer(is_critically_sampled=self.citically_sampled,
                                      # Input & output specifications.
                                      in_channels =in_channels,
                                      out_channels = self.lift_dim,
                                      in_size = in_size,
                                      out_size= in_size,
                                      in_sampling_rate=in_size,
                                      out_sampling_rate=in_size,
                                      in_cutoff = in_size/cutoff_den,
                                      out_cutoff= in_size/cutoff_den,
                                      in_half_width=  half_width_mult*in_size - in_size/cutoff_den,
                                      out_half_width= half_width_mult*in_size - in_size/cutoff_den,
                                      conv_kernel=conv_kernel,
                                      filter_size=filter_size,  
                                      lrelu_upsampling    = lrelu_upsampling,        
                                      use_radial_filters  =  self.radial
                                      ))

        ###################### Define U & D blocks ###################################################################

        self.cont_conv_layers = nn.ModuleList([(SynthesisLayer(is_critically_sampled=self.citically_sampled,
                                                               # Input & output specifications.
                                                               in_channels=self.feature_maps[i],
                                                               out_channels=self.out_feature_maps[i],
                                                               in_size=self.size[i],
                                                               out_size=self.size[i + 1],
                                                               in_sampling_rate=self.size[i],
                                                               out_sampling_rate=self.size[i + 1],
                                                               in_cutoff = self.cutoff[i],
                                                               out_cutoff= self.cutoff[i+1],
                                                               in_half_width= self.halfwidth[i],
                                                               out_half_width=self.halfwidth[i+1],
                                                               conv_kernel=conv_kernel,  
                                                               filter_size=filter_size, 
                                                               lrelu_upsampling    = lrelu_upsampling,
                                                               use_radial_filters  =  self.radial
                                                               ))
                                               for i in range(self.N_layers)])
                
        ###################### Define Invariant Blocks ##############################################################
        
        # Define cuttoff frequencies for invariant blocks
        self.cutoff_inv = np.zeros(len(self.size_inv))
        for i in range(len(self.size_inv)):
            self.cutoff_inv[i] = self.size_inv[i]/cutoff_den
        
        # Define halfwidth frequencies for invariant blocks
        self.halfwidth_inv = np.zeros(len(self.size_inv))
        for i in range(len(self.size_inv)):
            self.halfwidth_inv[i] = half_width_mult*self.size_inv[i] - self.size_inv[i]/cutoff_den
        
        self.cont_conv_layers_invariant = nn.ModuleList([(SynthesisLayer(is_critically_sampled=self.citically_sampled,
                                                                         # Input & output specifications.
                                                                         in_channels=self.feature_maps_invariant[i],
                                                                         out_channels=self.feature_maps_invariant[i],
                                                                         in_size=self.size_inv[i],
                                                                         out_size=self.size_inv[i],
                                                                         in_sampling_rate=self.size_inv[i],
                                                                         out_sampling_rate=self.size_inv[i],
                                                                         in_cutoff= self.cutoff_inv[i],
                                                                         out_cutoff= self.cutoff_inv[i],
                                                                         in_half_width= self.halfwidth_inv[i],
                                                                         out_half_width= self.halfwidth_inv[i],
                                                                         conv_kernel=conv_kernel,
                                                                         filter_size=filter_size,
                                                                         lrelu_upsampling    = lrelu_upsampling,
                                                                         use_radial_filters  = self.radial
                                                                         ))
                                                         for i in range(self.N_layers)])

        ###################### Define Batch Norm Layers (not middle ones) ###########################################
        
        # Do we use the batchnorm? In all our models, we USE it
        self.batch_ = batch_
        
        if self.batch_:
            self.batch_norm = [nn.BatchNorm2d(self.feature_maps[i])                                                               
                                               for i in range(self.N_layers//2+1)]
            self.batch_norm_inv = [nn.BatchNorm2d(self.feature_maps[i])                                                               
                                               for i in range(self.N_layers//2+1)]   
            
        ####################### Define ResNet Blocks ################################################################

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        self.resnet_blocks = []
        
        if self.batch_:
            self.res_batch_norm_in = []
        
        self.N_res = int(N_res)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers // 2 + 1):
            for i in range(N_res):
                self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[l],
                                                  in_size=self.size[l],
                                                  cutoff_den = cutoff_den,
                                                  radial     = radial,
                                                  conv_kernel=conv_kernel,
                                                  filter_size=filter_size,
                                                  lrelu_upsampling = lrelu_upsampling,
                                                  half_width_mult = half_width_mult,
                                                  length = self.length_res
                                                  ))
            
            if self.batch_:
                self.res_batch_norm_in.append(nn.BatchNorm2d(self.feature_maps[l]))
        
        ##################### Define BatchNorm Layers ###############################################################

        # We define all the other BatchNorm layers which we did not define before
        
        if self.batch_:
            for l in range(self.N_layers // 2 + 1,self.N_layers):
                self.batch_norm.append(nn.BatchNorm2d(self.feature_maps[l]))
                self.batch_norm_inv.append(nn.BatchNorm2d(self.feature_maps[l]))
            
        if self.batch_:
            self.res_batch_norm_in = torch.nn.Sequential(*self.res_batch_norm_in)
            self.batch_norm = torch.nn.Sequential(*self.batch_norm)
            self.batch_norm_inv = torch.nn.Sequential(*self.batch_norm_inv)
            
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)    

        #############################################################################################################

    def forward(self, x):
        
        #Lifting
        x = self.lift(x)
        skip = []
        
        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers // 2):
            
            # Apply (I) block
            if self.batch_ == True:
                x = self.batch_norm[i](x)
            x = self.cont_conv_layers_invariant[i](x)   
            
            # Apply ResNets
            if self.batch_ == True:
                y = self.res_batch_norm_in[i](x)   
            for j in range(self.N_res):
                y = self.resnet_blocks[i*self.N_res + j](x)
            skip.append(y)
            
            # Apply (D) layer
            if self.batch_ == True:
                x = self.batch_norm_inv[i](x)
            x = self.cont_conv_layers[i](x)

        #----------------------------------------------------------------------

        # Apply deepest ResNet
        if self.batch_ == True:
            x = self.res_batch_norm_in[-1](x)
        for i in range(self.N_res):
            x = self.resnet_blocks[-i-1](x)


        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers // 2, self.N_layers):
            
            # Apply (I) block
            if self.batch_ == True:
                x = self.batch_norm[i](x)
            x = self.cont_conv_layers_invariant[i](x)
            
            
            #(U) block
            if self.batch_ == True:               
                x = self.batch_norm_inv[i](x)
            x = self.cont_conv_layers[i](x)


            which = self.N_layers - i - 1 
            if i<self.N_layers-1:
                #U-Net concat.
                x = torch.cat((x,skip[which]),1)
            
            else:
                #Projection
                x = self.project(torch.cat((x,skip[which]), 1))

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
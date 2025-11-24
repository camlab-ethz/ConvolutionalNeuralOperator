# Implementation of the filters is borrowed from paper "Alias-Free Generative Adversarial Networks (StyleGAN3)" https://nvlabs.github.io/stylegan3/
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited. 

import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import einops
from training.filtered_networks import LReLu, LReLu_torch, LReLu_standard
from torch_utils.debug_tools import format_tensor_size
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

from DataLoaders.load_utils import _load_dataset

from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import copy

#--------------------------------------
# FiLM: Visual Reasoning with a General Conditioning Layer
# See https://arxiv.org/abs/1709.07871

class FILM(torch.nn.Module):
    def __init__(self, 
                channels,
                dim = [0,2,3],
                s = 128,
                intermediate = 128):
        super(FILM, self).__init__()
        self.channels = channels
        self.s = s
        
        self.inp2lat_sacale = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2scale = nn.Linear(in_features=intermediate, out_features=channels)

        self.inp2lat_bias = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2bias = nn.Linear(in_features=intermediate, out_features=channels)
        
        self.inp2lat_sacale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)
        self.lat2scale.bias.data.fill_(1)
        
        self.inp2lat_bias.weight.data.fill_(0)
        self.lat2bias.weight.data.fill_(0)
        self.lat2bias.bias.data.fill_(0)
        
        if dim == [0,2,3]:
            self.norm = nn.BatchNorm2d(channels)
        elif dim == [2,3]:
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif dim == [1,2,3]:
            self.norm = nn.LayerNorm([channels, s, s])
        else:
            self.norm = nn.Identity()
        
    def forward(self, x, timestep):
        
        x = self.norm(x)
        timestep = timestep.reshape(-1,1).type_as(x)
        scale     = self.lat2scale(self.inp2lat_sacale(timestep))
        bias      = self.lat2bias(self.inp2lat_bias(timestep))
        scale = scale.unsqueeze(2).unsqueeze(3)
        scale     = scale.expand_as(x)
        bias  = bias.unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale + bias 

#-------------------------------------------------
# One could add ViT into the bottleneck of the CNO
#-------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Takes a sequence of embedding of dimension dim
# 1. Applies LayerNorm
# 2. Applies linear layer dim -> 3x inner_dim
#                                NOTE: inner_dim = dim_head x heads
# 3. Applies attention
# 4. Projects inner -> dim

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Takes sequence of embeddings of dimension dim
# 1. Applies depth times:
#    a) Attention block: dim->dim (in the laast dimension)
#    b) MLP block:       dim->dim (in the laast dimension)
# 2. Applies LayerNorm
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Takes an image of size (n, c, h, w)
# Finds patch sizes (p_h, p_w) & number of patches (n_h, n_w)
# NOTE: It must hold that h%p_h == 0

# 1. Applies to_patch_embedding : 
#     a. (n, c, p_h*p1, p_w*p2) -> (n, n_h*n_w, p_h*p_w*c)
#     b. LayerNorm 
#     c. Linear embedding p_h*p_w*c -> dim
#     d. LayerNorm
# 2. Add positional embedding
# 3. Apply Transformer Block
# 4. TODO: Depatchify

class ViT(nn.Module):
    def __init__(self, 
                image_size, 
                patch_size, 
                dim, 
                depth, 
                heads, 
                mlp_dim = 256, 
                channels = 1, 
                dim_head = 32, 
                emb_dropout = 0.,):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.patch_to_image = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerBlock(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.patch_to_image(x)
        return x

#-------------------------------------------------
# CNO:
#-------------------------------------------------

#Depending on in_size, out_size, the CNOBlock can be:
#   -- (D) Block
#   -- (U) Block
#   -- (I) Block

class CNOBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                radial = False,
                batch_norm = True,
                activation = 'cno_lrelu',
                is_time = 4,
                nl_dim = [0],               
                time_steps = 5,
                lead_time_features = 512
                ):
        super(CNOBlock, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels #important for conditioning
        self.in_size      = in_size
        self.out_size     = out_size
        self.conv_kernel  = conv_kernel
        self.batch_norm   = batch_norm
        self.nl_dim       = nl_dim
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        
        self.in_halfwidth  =  half_width_mult*self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den
        
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels   = self.in_channels, 
                                            out_channels = self.out_channels, 
                                            kernel_size  = self.conv_kernel, 
                                            padding      = pad)
    
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()
        
        if activation == "cno_lrelu":
            self.activation  = LReLu(in_channels          = self.out_channels,
                                    out_channels          = self.out_channels,                   
                                    in_size               = self.in_size,                       
                                    out_size              = self.out_size,                       
                                    in_sampling_rate      = self.in_size,               
                                    out_sampling_rate     = self.out_size,             
                                    in_cutoff             = self.in_cutoff,                     
                                    out_cutoff            = self.out_cutoff,                  
                                    in_half_width         = self.in_halfwidth,                
                                    out_half_width        = self.out_halfwidth,              
                                    filter_size           = filter_size,       
                                    lrelu_upsampling      = lrelu_upsampling,
                                    is_critically_sampled = self.citically_sampled,
                                    use_radial_filters    = False)
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.out_channels,
                                            out_channels          = self.out_channels,                   
                                            in_size               = self.in_size,                       
                                            out_size              = self.out_size,                       
                                            in_sampling_rate      = self.in_size,               
                                            out_sampling_rate     = self.out_size)
        
        elif activation == "lrelu":
            self.activation  = LReLu_standard(in_channels          = self.out_channels,
                                            out_channels          = self.out_channels,                   
                                            in_size               = self.in_size,                       
                                            out_size              = self.out_size,                       
                                            in_sampling_rate      = self.in_size,               
                                            out_sampling_rate     = self.out_size)
        else:
            raise ValueError("Please specify different activation function")
        
        # Time conditioning:
        self.is_time = is_time
        self.time_steps = time_steps
    
        if is_time == 1 or is_time == True:
            self.time_steps = time_steps
            self.in_norm_conditiner = FILM(out_channels,
                                            dim = nl_dim,
                                            s = self.in_size)
            self.batch_norm = nn.Identity()
        
            
    def forward(self, x, time):
        x = self.convolution(x)
        x = self.batch_norm(x)
        if self.is_time == 1 or self.is_time == True:
            x = self.in_norm_conditiner(x, time)
        x = self.activation(x)
        return x

# Contains CNOBlock -> Convolution -> BN
class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                latent_dim = 64,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                radial = False,
                batch_norm = True,
                activation = 'cno_lrelu',
                is_time = True,
                time_steps = 5,
                lead_time_features = 512
                ):
        super(LiftProjectBlock, self).__init__()
    
        self.out_channels = out_channels #important for time conditioning
    
        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        in_size          = in_size,
                                        out_size         = out_size,
                                        cutoff_den       = cutoff_den,
                                        conv_kernel      = conv_kernel,
                                        filter_size      = filter_size,
                                        lrelu_upsampling = lrelu_upsampling,
                                        half_width_mult  = half_width_mult,
                                        radial           = radial,
                                        batch_norm       = batch_norm,
                                        activation       = activation,
                                        is_time = is_time,
                                        time_steps = time_steps,
                                        lead_time_features= lead_time_features)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels   = latent_dim, 
                                            out_channels = out_channels, 
                                            kernel_size  = conv_kernel, 
                                            stride       = 1, 
                                            padding      = pad)

    def forward(self, x, time):
        x = self.inter_CNOBlock(x, time)
        x = self.convolution(x)
        return x
        

# Residual Block containts:
# Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock(nn.Module):
    def __init__(self,
                channels,
                size,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                radial = False,
                batch_norm = True,
                activation = 'cno_lrelu',
                is_time = 4,
                nl_dim = [0], 
                time_steps = 5,
                lead_time_features = 512
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels #important for time conditioning
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.nl_dim = nl_dim
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels  = self.channels, 
                                            out_channels = self.channels, 
                                            kernel_size  = self.conv_kernel, 
                                            stride       = 1, 
                                            padding      = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                            kernel_size=self.conv_kernel, stride = 1, 
                                            padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()
        
        if activation == "cno_lrelu":

            self.activation  = LReLu(in_channels          = self.channels, #In _channels is not used in these settings
                                    out_channels          = self.channels,                   
                                    in_size               = self.size,                       
                                    out_size              = self.size,                       
                                    in_sampling_rate      = self.size,               
                                    out_sampling_rate     = self.size,             
                                    in_cutoff             = self.cutoff,                     
                                    out_cutoff            = self.cutoff,                  
                                    in_half_width         = self.halfwidth,                
                                    out_half_width        = self.halfwidth,              
                                    filter_size           = filter_size,       
                                    lrelu_upsampling      = lrelu_upsampling,
                                    is_critically_sampled = self.citically_sampled,
                                    use_radial_filters    = False)
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        
        elif activation == "lrelu":

            self.activation = LReLu_standard(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")
        
        # Time conditioning:
        self.is_time = is_time
        if self.is_time==1 or self.is_time == True:
            self.time_steps = time_steps
            self.in_norm_conditiner1 = FILM(channels,
                                            dim = nl_dim,
                                            s = self.size)
            self.in_norm_conditiner2 = FILM(channels,
                                            dim = nl_dim,
                                            s = self.size)
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()
        


    def forward(self, x, time):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        if self.is_time==1 or self.is_time == True:
            out = self.in_norm_conditiner1(out, time)
        out = self.activation(out)
        out = self.convolution2(out)

        out = self.batch_norm2(out)
        if self.is_time==1 or self.is_time == True:
            out = self.in_norm_conditiner2(out, time)

        x = x + out
        del out
        
        return x


# CNO NETWORK:
class CNO_time(pl.LightningModule):
    def __init__(self,  
                in_dim,                    # Number of input channels.
                in_size,                   # Input spatial size
                N_layers,                  # Number of (D) or (U) blocks in the network
                N_res = 1,                 # Number of (R) blocks per level (except the neck)
                N_res_neck = 6,            # Number of (R) blocks in the neck
                channel_multiplier = 32,   # How the number of channels evolve?
                conv_kernel=3,             # Size of all the kernels
                cutoff_den = 2.0001,       # Filter property 1.
                filter_size=6,             # Filter property 2.
                lrelu_upsampling = 2,      # Filter property 3.
                half_width_mult  = 0.8,    # Filter property 4.
                radial = False,            # Filter property 5. Is filter radial?
                batch_norm = True,         # Add BN? We do not add BN in lifting/projection layer
                out_dim = 1,               # Target dimension
                out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                latent_lift_proj_dim = 64, # Intermediate latent dimension in the lifting/projection layer
                add_inv = True,            # Add invariant block (I) after the intermediate connections?
                activation = 'cno_lrelu',  # Activation function can be 'cno_lrelu' or 'lrelu'
                
                is_att = False,
                patch_size = 1,
                dim_multiplier = 1.0,
                depth = 4,
                heads = 4,
                dim_head_multiplier = 1.0,
                mlp_dim_multiplier = 1.0,
                emb_dropout = 0.,
                
                time_steps = 5,
                is_time = 4,
                nl_dim  = [1],
                
                p_loss = 1,
                lr = 0.0005,
                batch_size = 60,
                weight_decay = 1e-6,
                scheduler_step = 15,
                scheduler_gamma = 0.98,
                loader_dictionary = dict(),
                ):
        
        super(CNO_time, self).__init__()

        ### Define the parameters & specifications ###        
        
        # Number od (D) & (U) Blocks
        self.N_layers = int(N_layers)
        
        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2         
        self.in_dim = in_dim
        self.out_dim = out_dim   
        
        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        
        
        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True
        
        ### Define evolution of the number features ###

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)
        
        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])
        
        _lead_time_features = max(self.inv_features)
        if self.encoder_features[-1]>_lead_time_features:
            _lead_time_features = self.encoder_features[-1]
        _lead_time_features = 2*_lead_time_features
        
        ### Define evolution of sampling rates ###
        
        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
        
        ### Define Projection & Lift ###
    
        self.lift = LiftProjectBlock(in_channels  = in_dim,
                                    out_channels = self.encoder_features[0],
                                    in_size      = in_size,
                                    out_size     = self.encoder_sizes[0],
                                    latent_dim   = latent_lift_proj_dim,
                                    cutoff_den   = cutoff_den,
                                    conv_kernel  = conv_kernel,
                                    filter_size  = filter_size,
                                    lrelu_upsampling  = lrelu_upsampling,
                                    half_width_mult   = half_width_mult,
                                    radial            = radial,
                                    batch_norm        = False,
                                    activation = activation,
                                    is_time = False,
                                    time_steps = time_steps,
                                    lead_time_features = _lead_time_features)
        
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
        
        self.project = LiftProjectBlock(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = out_dim,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        radial            = radial,
                                        batch_norm        = False,
                                        activation = activation,
                                        is_time = False,
                                        time_steps = time_steps,
                                        lead_time_features = _lead_time_features)
        
        ### Define U & D blocks ###

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))                                  
                                                for i in range(self.N_layers)])
        
        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))                                  
                                                for i in range(self.N_layers + 1)])
        
        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))                                  
                                                for i in range(self.N_layers)])
        
        self.decoder_inv    = nn.ModuleList([(CNOBlock(in_channels  =  self.inv_features[i],
                                                        out_channels = self.inv_features[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        is_time=is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))                               
                                                for i in range(self.N_layers + 1)])

        #### Define ResNets Blocks ###

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(ResidualBlock(channels = self.encoder_features[l],
                                                    size     = self.encoder_sizes[l],
                                                    cutoff_den = cutoff_den,
                                                    conv_kernel = conv_kernel,
                                                    filter_size = filter_size,
                                                    lrelu_upsampling = lrelu_upsampling,
                                                    half_width_mult  = half_width_mult,
                                                    radial = radial,
                                                    batch_norm = batch_norm,
                                                    activation = activation,
                                                    is_time=is_time,
                                                    nl_dim = nl_dim,
                                                    time_steps = time_steps,
                                                    lead_time_features = _lead_time_features))  
        for i in range(self.N_res_neck):
            self.res_nets.append(ResidualBlock(channels = self.encoder_features[self.N_layers],
                                                size     = self.encoder_sizes[self.N_layers],
                                                cutoff_den = cutoff_den,
                                                conv_kernel = conv_kernel,
                                                filter_size = filter_size,
                                                lrelu_upsampling = lrelu_upsampling,
                                                half_width_mult  = half_width_mult,
                                                radial = radial,
                                                batch_norm = batch_norm,
                                                activation = activation,
                                                is_time=is_time,
                                                nl_dim = nl_dim,
                                                time_steps = time_steps,
                                                lead_time_features = _lead_time_features))
        self.res_nets = torch.nn.Sequential(*self.res_nets)    

        ### Transformer bn ###
        
        if not is_att:
            self.transformer = nn.Identity()
        else:
            _dim = int(dim_multiplier*patch_size**2*self.encoder_features[self.N_layers])
            self.transformer  = ViT(image_size  = self.encoder_sizes[self.N_layers], 
                                    patch_size  = patch_size, 
                                    dim         = _dim, 
                                    depth       = depth, 
                                    heads       = heads, 
                                    mlp_dim     = int(mlp_dim_multiplier * _dim),
                                    channels    = self.encoder_features[self.N_layers], 
                                    dim_head    = int(dim_head_multiplier * _dim),
                                    emb_dropout = emb_dropout)
    
        #### Training parameters ####
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.loader_dictionary = loader_dictionary # IMPORTANT -- Experiment and Training parameters
        
        self.validation_step_outputs = []
        self.validation_times = []
        
        self.best_val_loss_mean = 1000
        self.best_val_loss_median = 1000
        self.best_val_loss_mean_last = 1000
        self.best_val_loss_median_last = 1000
        
        self.validation_errs  = dict()
        self.validation_times = dict() 

         
        # If we traing the model to predict different physical quantities (velocity + pressure + ...)
        # For example, if the variables are [rho, vx, vy, p], then "separate_dim" should be [1,2,1]
        # 2 means that vx and vy are grouped together!
        
        if ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"]:
            assert "separate_dim" in self.loader_dictionary
            print(self.loader_dictionary["separate_dim"], "separate_dim")
            self.validation_errs_sep = dict()
            self.test_errs_sep = dict()
            
    def forward(self, x, time):
        
        #Execute Lift ---------------------------------------------------------
        x = self.lift(x, time)
        skip = []
        
        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            #Apply ResNet & save the result
            #y = x
            for j in range(self.N_res):
                x = self.res_nets[i*self.N_res + j](x, time)
            skip.append(x)
            
            # Apply (D) block
            x = self.encoder[i](x, time)   
        
        # BN: -----------------------------------------------------------------
        
        # Apply transformer (if is_att == True)
        if hasattr(self, 'transformer'):
            x = self.transformer(x)
        
        # Apply the deepest ResNet (bottle neck)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x, time)
        
        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x, time) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i], time)),1)
            
            if self.add_inv:
                x = self.decoder_inv[i](x, time)
            # Apply (U) block
            x = self.decoder[i](x, time)
        
        # Cat & Execute Projetion ---------------------------------------------
        
        x = torch.cat((x, self.ED_expansion[0](skip[0], time)),1)
        x = self.project(x, time)
        return x
    
    
    
    def training_step(self, batch):
        
        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list             
        
        #---------
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        #---------
        if "is_masked" in self.loader_dictionary:
            is_masked = self.loader_dictionary["is_masked"] is not None
        else:
            is_masked = False
        
        if not is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch
        
        # Predict:
        output_pred_batch = self(input_batch, t_batch)
        #---------
        
        # If airfoil, mask it
        which = self.loader_dictionary["which"]
        if "airfoil" in which:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0
            
        #---------------
        # Compute the loss - Relative L1
        #---------------
        
        if not is_separate:
            loss = nn.L1Loss()(output_batch, output_pred_batch) / nn.L1Loss()(torch.zeros_like(output_batch), output_batch)
 
        else:
            # How are the variables separated?
            diff = [0, separate_dim[0]]
            for i in range(1,len(separate_dim)):
                diff.append(diff[-1]+separate_dim[i])
            self.num_separate = len(diff)-1 

            loss = 0.0
            if not is_masked:
                # Compute the loss over each block in 'separated' output
                weight = 1.0/self.num_separate
                for i in range(self.num_separate):
                    loss = loss + weight*nn.L1Loss()(output_pred_batch[:,diff[i]:diff[i+1]], output_batch[:,diff[i]:diff[i+1]])/ (nn.L1Loss()(output_batch[:,diff[i]:diff[i+1]],torch.zeros_like(output_batch[:,diff[i]:diff[i+1]])) + 1e-10)

            else:
                
                # Mask and compute the loss
                for i in range(self.num_separate):
                    mask = masked_dim[:,diff[i]:diff[i+1]]
                    mask = mask.unsqueeze(-1).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
                    output_pred_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
                    output_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0

                    loss = loss + nn.L1Loss()(output_pred_batch[:,diff[i]:diff[i+1]], output_batch[:,diff[i]:diff[i+1]])/ nn.L1Loss()(output_batch[:,diff[i]:diff[i+1]],torch.zeros_like(output_batch[:,diff[i]:diff[i+1]]) + 1e-10) 
           
        
        self.log("loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        
        # Are we in the FT regime?
        if "fine_tuned" in self.loader_dictionary and self.loader_dictionary["fine_tuned"]:
            
            print("=========================")
            print("Configure Optimizers - FT")
            print("=========================")
            assert hasattr(self, 'lr_emb')
            assert hasattr(self, 'lr_norm')

            params_1 = [param for name, param in self.named_parameters() if ("project" not in name) and ("lift" not in name) and ("in_norm_conditiner" not in name)]

            params_2 = [param for name, param in self.named_parameters() if (("project" in name) or ("lift" in name)) and ("in_norm_conditiner" not in name)]
            
            params_3 = [param for name, param in self.named_parameters() if ("in_norm_conditiner" in name) and ("project" not in name) and ("lift" not in name)]

            optimizer = torch.optim.AdamW([{'params': params_1},
                                           {'params': params_2,
                                            'lr': self.lr_emb},
                                           {'params': params_3,
                                            'lr': self.lr_norm}],
                                           lr=self.lr, weight_decay = self.loader_dictionary["weight_decay"])     
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
         #Scheduler does not depend on the regime
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] 
    
    def train_dataloader(self):
        
        which = self.loader_dictionary["which"]      # which benchmark
        
        if which == "eul_ns_mix1":
            train_dataset1 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_riemann", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            
            train_dataset2 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_riemann_cur", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            
            train_dataset3 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_gauss", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            
            
            train_dataset4 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_kh", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            
            
            train_dataset5 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "ns_gauss", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 0.0])
            
            train_dataset6 = _load_dataset(dic = self.loader_dictionary, 
                                          which = "ns_sin", 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 0.0])
            
            train_datasets = [train_dataset1, train_dataset2, train_dataset3,
                              train_dataset4, train_dataset5, train_dataset6]
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
        else:
            is_masked =  "is_masked" in self.loader_dictionary and self.loader_dictionary["is_masked"] is not None
            if is_masked:  
                if which[:2] == "ns":
                    mask = [1.0, 1.0, 1.0, 0.0]
                elif which[:3] == "eul":
                    mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = [1.0, 1.0, 1.0, 1.0]
            else:
                mask = None
                
            
            train_dataset = _load_dataset(dic = self.loader_dictionary, 
                                          which = which, 
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = mask)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
        return train_loader
        
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        
        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list  
        
        #---------
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        #---------
        if "is_masked" in self.loader_dictionary:
            is_masked = self.loader_dictionary["is_masked"] is not None
        else:
            is_masked = False
        
        if not is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch
        
        # Predict:
        output_pred_batch = self(input_batch, t_batch)
        #---------
        
        # If airfoil, mask it
        which = self.loader_dictionary["which"]
        if "airfoil" in which:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0
            
        #---------------
        # Compute the loss
        #---------------
        if not is_masked:
            loss = (torch.mean(abs(output_pred_batch - output_batch), (-3, -2, -1)) / (torch.mean(abs(output_batch), (-3, -2, -1))+ 1e-10))* 100
        else:
            mask = masked_dim.unsqueeze(-1).unsqueeze(-1).expand(masked_dim.shape[0], masked_dim.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
            output_pred_batch[mask==0.0] = 1.0
            output_batch[mask==0.0] = 1.0

            
            loss = (torch.mean(abs(output_pred_batch - output_batch), (-3, -2, -1)) / (torch.mean(abs(output_batch), (-3, -2, -1)) + 1e-10))* 100
            
        #---------------
        # If it is separate - compute loss over the dimension
        #---------------
        if is_separate:

            diff = [0, self.loader_dictionary["separate_dim"][0]]
            for i in range(1,len(self.loader_dictionary["separate_dim"])):
                diff.append(diff[-1]+self.loader_dictionary["separate_dim"][i])
            self.num_separate = len(diff)-1 
            
            # Masked?
            if not is_masked:

                loss_sep = []
                for i in range(self.num_separate):
                    _loss = (torch.mean(abs(output_pred_batch[:,diff[i]:diff[i+1]] - output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1)) / (torch.mean(abs(output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1))+ 1e-10))* 100
                    loss_sep.append(_loss)

            else:
                loss_sep = []
                for i in range(self.num_separate):
                    mask = masked_dim[:,diff[i]:diff[i+1]]
                    mask = mask.unsqueeze(-1).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
                    output_pred_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
                    output_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0

                    loss_sep.append((torch.mean(abs(output_pred_batch[:,diff[i]:diff[i+1]] - output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1)) / (torch.mean(abs(output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1))+1e-10))* 100)   
        
        #---------------
        # Save validation errs:
        #---------------
        if batch_idx==0:
            self.validation_times[str(dataloader_idx)] = t_batch
            self.validation_errs[str(dataloader_idx)] = loss
            
            if is_separate:
                self.validation_errs_sep[str(dataloader_idx)] = []
                for i in range(self.num_separate):
                    self.validation_errs_sep[str(dataloader_idx)].append(loss_sep[i])
                        
        else:
            
            self.validation_times[str(dataloader_idx)] = torch.cat((self.validation_times[str(dataloader_idx)], t_batch))
            self.validation_errs[str(dataloader_idx)] = torch.cat((self.validation_errs[str(dataloader_idx)], loss))
                   
            if is_separate:
                for i in range(self.num_separate):
                    self.validation_errs_sep[str(dataloader_idx)][i] = torch.cat((self.validation_errs_sep[str(dataloader_idx)][i], loss_sep[i]))
                
        return loss
        
        
    def val_dataloader(self):
        which = self.loader_dictionary["which"]      # which benchmark
        
        val_datasets = []
        num_datasets = 1
        num_out      = 0

        if which == "eul_ns_mix1":
            val_dataset1  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_riemann", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            val_dataset2  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_riemann_cur", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            val_dataset3  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_gauss", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            val_dataset4  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "eul_kh", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 1.0])
            
            val_dataset5  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "ns_gauss", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 0.0])
            val_dataset6  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = "ns_sin", 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = [1.0, 1.0, 1.0, 0.0])
 
            val_datasets = [val_dataset1, val_dataset2, val_dataset3,
                            val_dataset4, val_dataset5, val_dataset6]
            num_datasets = 6
            num_out = 0
            
            self.val_labels = ["CE_Ri_", "CE_RiCu_", "CE_Gau_", "CE_KH_", "IC_Gau_", "IC_Sin_"]
            
            val_loaders = []
            for dataset in val_datasets:
                val_loaders.append(DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=6))
                
        else:
            is_masked = "is_masked" in self.loader_dictionary and self.loader_dictionary["is_masked"] is not None
            if is_masked:
                if which[:2] == "ns":
                    mask = [1.0, 1.0, 1.0, 0.0]
                elif which[:3] == "eul":
                    mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = [1.0, 1.0, 1.0, 1.0]
            else:
                mask = None
            
            val_dataset  =  _load_dataset(dic = self.loader_dictionary, 
                                          which = which, 
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = mask)
            self.val_labels =[which+"_"]
            val_loaders = [DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)]
        
        self.num_validation_loaders = num_datasets
        self.num_out_loaders        = num_out
        
        return val_loaders
        
        
    def on_validation_epoch_end(self):

        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list
        
        # What to do with all the loaders?
        for dataloader_idx in range(self.num_validation_loaders + self.num_out_loaders):
            
            _stack = self.validation_errs[str(dataloader_idx)]
            
            if is_separate:
                _stack_sep = self.validation_errs_sep[str(dataloader_idx)] 
                
            if dataloader_idx == 0:
                _stack_all = _stack
                
            elif dataloader_idx < self.num_validation_loaders:
                _stack_all = torch.cat((_stack_all, _stack))
            
            idx_label = self.val_labels[dataloader_idx]
            median_loss = torch.median(_stack).item()
            mean_loss = torch.mean(_stack).item()
                        
            prog_bar = True
            
            if self.num_validation_loaders + self.num_out_loaders > 1:
                self.log(idx_label + "med_val_l", median_loss, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)
                self.log(idx_label + "mean_val_l",  mean_loss, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)

            if is_separate:
                
                for i in range(self.num_separate):
                    median_loss_s = torch.median(_stack_sep[i]).item()
                    mean_loss_s = torch.mean(_stack_sep[i]).item()

                    self.log(idx_label+"mean_val_" + str(i),  mean_loss_s, on_step=False, on_epoch=True,sync_dist=True)
                    self.log(idx_label+"med_val_"  + str(i),  median_loss_s, on_step=False, on_epoch=True,sync_dist=True)
             
        
        
            
        median_loss = torch.median(_stack_all).item()
        mean_loss = torch.mean(_stack_all).item() 
        
        self.log("med_val_l", median_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("mean_val_l",  mean_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        # Save the best loss
        if mean_loss<self.best_val_loss_mean:
            self.best_val_loss_mean = mean_loss
            self.best_val_loss_median = median_loss
        
        self.log("best_mean_val_loss",self.best_val_loss_mean,on_step=False, on_epoch=True,sync_dist=True)
        self.log("best_med_val_loss",self.best_val_loss_median,on_step=False, on_epoch=True,sync_dist=True)
                
        return {"med_val_l": median_loss, "mean_val_l": mean_loss,} 
    
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
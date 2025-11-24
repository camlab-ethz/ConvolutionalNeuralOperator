import torch.nn as nn
import torch
import os

from CNO_timeModule_CIN import LiftProjectBlock, FILM, CNOBlock


#--------------------------
# Finetune Lift & Project
#--------------------------

class FT_Lift(torch.nn.Module):
    def __init__(self, 
                in_channels,
                latent_channels,
                model,
                pre_lift = False):
        super(FT_Lift, self).__init__()
        
        self.latent_channels = latent_channels
        self.pre_lift = pre_lift
        
        if pre_lift:
            self.linear1 = torch.nn.Conv2d(in_channels     = in_channels, 
                                            out_channels   = 128, 
                                            kernel_size    = 1,
                                            padding        = 0)

            self.linear2 = torch.nn.Conv2d(in_channels     = 128, 
                                            out_channels   = latent_channels, 
                                            kernel_size    = 1,
                                            padding        = 0)
        else:
            self.linear1 = torch.nn.Conv2d(in_channels     = in_channels, 
                                            out_channels   = in_channels, 
                                            kernel_size    = 1,
                                            padding        = 0)
            self.linear2 = nn.Identity()
            
        self.lift    = model.lift
        
        
    def forward(self, x, timestep):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.lift(x, timestep)
        return x

class FT_Project(torch.nn.Module):
    def __init__(self, 
                new_out_dim,
                model,
                init_project = False):
        super(FT_Project, self).__init__()
        
        if init_project:
            self.in_channels = model.encoder_features[0] + model.decoder_features_out[-1]
            self.project = LiftProjectBlock(in_channels  = self.in_channels,
                                    out_channels = new_out_dim,
                                    in_size      = model.decoder_sizes[-1],
                                    out_size     = 128,
                                    latent_dim   = 64,
                                    batch_norm        = False,
                                    is_time = False)
            self.post_linear1 = nn.Identity()
            self.post_linear2 = nn.Identity()
        else:
            self.project    = model.project
            self.post_linear1 = torch.nn.Conv2d(in_channels     = new_out_dim, 
                                               out_channels    = 128, 
                                               kernel_size     = 1,
                                               padding         = 0)
            self.post_linear2 = torch.nn.Conv2d(in_channels     = 128, 
                                               out_channels    = new_out_dim, 
                                               kernel_size     = 1,
                                               padding         = 0)
        
    def forward(self, x, timestep):
        x = self.project(x, timestep)
        x = self.post_linear1(x) 
        x = self.post_linear2(x) 

        return x
    

def initialize_FT(model,
                   old_in_dim,
                   new_in_dim,
                   new_out_dim,
                   old_out_dim):

    pre_lift = old_in_dim != new_in_dim
    new_lift = FT_Lift(in_channels = new_in_dim,
                       latent_channels = old_in_dim,
                       model = model,
                       pre_lift = pre_lift)
    model.lift = new_lift
    
    
    init_project = new_out_dim != old_out_dim
    model.project = FT_Project(new_out_dim = new_out_dim,
                                model = model,
                                init_project = init_project)
    
    print(init_project, "init_project", pre_lift, "pre_lift")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model
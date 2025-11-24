import torch.nn as nn
import torch
import os

from CNO_timeModule_CIN import CNO_time
from CNO_timeModule_CIN import CNO_time, LiftProjectBlock

from test_and_fine_tune_utils.test_utils import _load_dict



def _find_model(folder, 
                label = "123"):        
    _potential_files = os.listdir(folder + "/model"+label)
    for file in _potential_files:
        if "epoch" in file:
            model_file = file
            break
    return model_file

def _initialize_model(loader_dict,
                     in_dim,
                     out_dim):
    
    model   = CNO_time(in_dim =  in_dim,               
                        in_size  = 128,                  
                        N_layers = loader_dict["N_layers"],                
                        N_res =  loader_dict["N_res"], 
                        N_res_neck = loader_dict["N_res_neck"],           
                        channel_multiplier = loader_dict["channel_multiplier"],  
                        batch_norm = loader_dict["batch_norm"],        
                        out_dim = out_dim,            
                        activation = loader_dict["activation"], 
                        time_steps = loader_dict["time_steps"],
                        is_time = loader_dict["is_time"],
                        p_loss = loader_dict["exp"],           
                        lr = loader_dict["learning_rate"],
                        batch_size = loader_dict["batch_size"],
                        weight_decay = loader_dict["weight_decay"],
                        loader_dictionary = loader_dict,
                        nl_dim= loader_dict["nl_dim"],
                        is_att = loader_dict["is_att"],
                        patch_size = loader_dict["patch_size"],
                        dim_multiplier = loader_dict["dim_multiplier"],
                        depth = loader_dict["depth"],
                        heads = loader_dict["heads"],
                        dim_head_multiplier = loader_dict["dim_head_multiplier"],
                        mlp_dim_multiplier = loader_dict["mlp_dim_multiplier"],
                        emb_dropout = loader_dict["emb_dropout"])
    return model

def load_model(folder, 
               which_example, 
               in_dim,
               out_dim,
               steps = 7,
               is_masked = None,
               label = "123"):
    train_file  = folder + "/training_properties.txt"
    net_file    = folder + "/net_architecture.txt"
    loader_dict = _load_dict(files = [train_file, net_file], 
                             which_example = which_example, 
                             steps = steps,
                             is_masked = is_masked)
    _model_file = _find_model(folder, label)
    model_file = folder + "/model"+ label+"/" + _model_file
    model = _initialize_model(loader_dict,
                             in_dim = in_dim,
                             out_dim = out_dim)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_file, map_location = device)
    model.load_state_dict(checkpoint["state_dict"])        
    model = model.to(device)
    
    return model, loader_dict

def allowed_transitions(one_all, time_steps = 7):
    _allowed = []
    if one_all == 'all':
        for t in range(0,time_steps+1):
            _allowed.append(t)
    elif one_all == 'one':
        _allowed = [1]
    
    return _allowed

def initialize_emb(model,
                   new_in_dim,
                   new_out_dim):
    new_lift     = LiftProjectBlock(in_channels = new_in_dim,
                                out_channels = model.encoder_features[0],
                                in_size      = 128,
                                out_size     = model.encoder_sizes[0],
                                latent_dim   = 64,
                                batch_norm   = False,
                                is_time      = False)
    
    new_project  = LiftProjectBlock(in_channels  = model.encoder_features[0] + model.decoder_features_out[-1],
                                    out_channels = new_out_dim,
                                    in_size      = model.decoder_sizes[-1],
                                    out_size     = 128,
                                    latent_dim   = 64,
                                    batch_norm        = False,
                                    is_time = False)
    model.lift = new_lift
    model.project = new_project
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model

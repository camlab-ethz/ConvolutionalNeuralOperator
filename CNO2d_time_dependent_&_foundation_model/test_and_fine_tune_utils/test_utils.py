from CNO_timeModule_CIN import CNO_time

import torch.nn as nn
import torch

from test_and_fine_tune_utils.fine_tune_lift import initialize_FT

from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import copy
import numpy as np

from DataLoaders.load_utils import _load_dataset

#------------------------------------------------
# Provide a current dictionary and a file to load
# It is a bit messy, but it works :)

def _load_dict(files, 
               which_example,
               steps = 7,
               is_masked = None):
    d = dict()
    
    d["which"] = which_example
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        f.close()
        
        for line in lines:
            l = line.strip().split(",")
            if l[0] in ["epochs", "batch_size", "exp", "training_samples", "time_steps", "dt"]:
                value = int(float(l[1]))
            elif l[0] == "nl_dim":
                s = ""
                for a in l[1:]:
                    if a[0]=="\"":
                        a = a.strip(" ")
                        s+=a[-1]
                    elif a[-1] == "\"":
                        a = a.strip(" ")
                        s+=a[0]
                    else:
                        s+=a.strip(" ")
                value = s
                S = value.split(", ")
                string = ""
                for s in S:
                    string+=s
                value = string
                
            elif "." in l[1]:
                value = float(l[1])
            elif "False" in l[1]:
                value = False
            elif "True" in l[1]:
                value = True
            elif "relu" in l[1]:
                value = l[1]
            elif "e-" in l[1]:
                value = float(l[1])
            elif "allowed" in l[0]:
                value = l[1]
            else:
                value = int(l[1])
            d[l[0]] = value
    
    #------------------
    
    if "batch_norm" in d:
        d["batch_norm"] = True if d["batch_norm"] in [1, True] else False
    if "time_input" in d:
        d["time_input"] = True if d["time_input"] in [1, True] else False
    if "cluster" in d:
        d["cluster"]    = True if d["cluster"]    in [1, True] else False
    
    #------------------
    
    if "nl_dim" in d:
        if d["nl_dim"] in ["023"]:
            d["nl_dim"] = [0,2,3]
        elif d["nl_dim"] in ["123"]:
            d["nl_dim"] = [1,2,3]
        elif d["nl_dim"] in ["23"]:
            d["nl_dim"] = [2,3]
    
    #------------------
    
    if "is_att" in d:
        d["is_att"] = True if d["is_att"] in [1, True] else False
    else:
        d["is_att"] = False
        d["patch_size"] = None
        d["dim_multiplier"] = None
        d["depth"] = None
        d["heads"] = None
        d["dim_head_multiplier"] = None
        d["mlp_dim_multiplier"] = None
        d["emb_dropout"] = None
    
    #------------------
    
    d["masked_input"] = None
    if is_masked:
        d["is_masked"] = is_masked 
    else:
        d["is_masked"] = None 
    
    d["time_steps"] = steps
    
    #------------------
    
    if "poisson" in which_example:
        d["in_dim"] = 1
        d["out_dim"] = 1
        d["nmax"] = 20000
        d["step_max"] = 1
        d["time_input"] = False
        d["dt"] = 0
    
    elif which_example == "helmholtz":
        
        d["in_dim"] = 2
        d["out_dim"] = 1
        d["nmax"] = 19675
        d["step_max"] = 1
        d["time_input"] = False
        d["dt"] = 0
    
    elif which_example in ["allen_cahn"]:
        d["in_dim"] = 2 if d["time_input"] == True else 1
        d["out_dim"] = 1
        d["nmax"] = 15000
        d["step_max"] = 19
    
    elif which_example in ["wave_seismic", "wave_gauss"]:
        d["in_dim"] = 3 if d["time_input"] == True else 2
        d["out_dim"] = 2
        d["separate"] = True
        d["separate_dim"] = [1,1]
        
        d["nmax"] = 10512
        if "seismic" in which_example:
            d["step_max"] = 20
        else:
            d["step_max"] = 15

    elif which_example in ["ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear"]:

        d["nmax"] = 20000
        d["step_max"] = 20
        
        if d["is_masked"] is not None:
            d["in_dim"] = 5 if d["time_input"] == True else 4
            d["out_dim"] = 4
            d["separate"] = True
            d["separate_dim"] = [1,2,1]
            d["masked_input"] = [1.0, 1.0, 1.0, 0.0]
        else:
            d["in_dim"] = 3 if d["time_input"] == True else 2
            d["out_dim"] = 2
    
    elif which_example in ["ns_pwc_t"]:
        d["in_dim"] = 4 if d["time_input"] == True else 3
        d["out_dim"] = 3
        d["nmax"] = 20000
        d["step_max"] = 20
        d["separate"] = True
        d["separate_dim"] = [2,1]
    
    elif which_example in ["eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"]:
        d["in_dim"] = 5 if d["time_input"] == True else 4
        d["out_dim"] = 4
        d["nmax"] = 10000
        d["step_max"] = 20
        d["separate"] = True
        d["separate_dim"] = [1,2,1]
        
    elif which_example in ["rich_mesh"]:     
        d["nmax"] = 1260
        d["step_max"] = 20
        d["separate"] = True        
        d["separate_dim"] = [1,2,1]
        d["in_dim"] = 5 if d["time_input"] == True else 4
        d["out_dim"] = 4
    
    elif which_example in ["rayl_tayl"]:     
        d["nmax"] = 1260
        d["step_max"] = 10
        d["separate"] = True
        d["dt"] = 1
        d["separate_dim"] = [1,2,1,1]
        d["in_dim"] = 6 if d["time_input"] == True else 5
        d["out_dim"] = 5


    elif which_example in ["kolmogorov"]:
        d["in_dim"] = 4 if d["time_input"] == True else 3
        d["out_dim"] = 3
        d["separate"] = True
        d["separate_dim"] = [2, 1]
        d["nmax"] = 20000
        d["step_max"] = 20
    
    elif which_example in ["airfoil"]:
        d["in_dim"] = 1
        d["out_dim"] = 1
        d["nmax"] = 10869
        d["step_max"] = 1
        d["time_input"] = False
        d["dt"] = 0

    else:
        raise ValueError("Please specify different benchmark")
    
    
    assert "in_dim" in d
    assert "out_dim" in d
    
    
    _allowed = []
    print("STEPS", d["time_steps"])
    d["allowed"] = 'all'
    if d['allowed'] == 'all':
        for t in range(1,d["time_steps"]+1):
            _allowed.append(t)
    elif d['allowed'] == 'one':
        _allowed = [1]
    d["allowed_tran"] = _allowed
    
    return d

#------------------------------------------------
#Initialize the model using loading_dict

def _initialize_model(loader_dict,
                      diff_embedding = False,
                      old_in_dim  = 5,
                      new_in_dim  = 5,
                      old_out_dim = 4,
                      new_out_dim = 4):
    
    if diff_embedding:
        _in_dim = old_in_dim
        _out_dim = old_out_dim
    else:
        _in_dim = loader_dict["in_dim"]
        _out_dim = loader_dict["out_dim"]
        
    model   = CNO_time(in_dim =  _in_dim,
                       out_dim = _out_dim,
                        in_size  = 128,                  
                        N_layers = loader_dict["N_layers"],                
                        N_res =  loader_dict["N_res"], 
                        N_res_neck = loader_dict["N_res_neck"],           
                        channel_multiplier = loader_dict["channel_multiplier"],  
                        batch_norm = loader_dict["batch_norm"],            
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
                        emb_dropout = loader_dict["emb_dropout"],
                        )
    if diff_embedding:
        model = initialize_FT(model = model,
                               old_in_dim = 5,
                               new_in_dim = new_in_dim,
                               new_out_dim = new_out_dim,
                               old_out_dim = 4)
    return model


#------------------------------------------------
# Provide the dataloader

def _load_data(which, 
                loader_dict,
                batch_size,
                fix_input_to_time_step = None,
                which_loader = "test",
                in_dist = True, 
                cluster = False,
                seq_mode = False): #For AR finetuning
    
    if "train" in which_loader[:5]:
        samples = loader_dict["num_samples"]
    else:
        samples = 1
    loader_dict['num_samples'] = samples
    
    if "is_masked" in loader_dict:
        masked_input = loader_dict["masked_input"]
    else:
        masked_input = None
    
    loader_dict["normalize"] = True
    
    test_dataset = _load_dataset(dic = loader_dict, 
                                 which = which, 
                                 which_loader = which_loader,
                                 in_dim = None,
                                 out_dim = None,
                                 masked_input = masked_input,
                                 fix_input_to_time_step = fix_input_to_time_step)
    
    if "train" in which_loader:
        shuffle = True
    else:
        shuffle = False
    
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader



# List all the models in a training directory (e.g. Light_ShearLayer_Scaling_l5t2_is0bn1_1)
# Models are listed in a form of tuples: 
# ('Scaling_NS/Light_ShearLayer_Scaling_l5t2_is0bn1_1/2500Setup_2', 'epoch=1919-step=40320.ckpt')
def _list_models(folders, 
                 flags = []):
    model_dirs = []
        
    dirs = os.listdir(folders)
    for item in dirs:
        if os.path.isdir(folders+"/"+item):# and "4096" not in item and "2048" not in item:
            model_dirs.append(folders+"/"+item)
    
    models = []
    for dir in model_dirs:
        
        model_dir = dir + "/model123"
        _potential_files = os.listdir(model_dir)
        for file in _potential_files:
            _pass = True
            _pass = _pass and "epoch" in file
            
            for flag in flags:
                _pass = _pass and flag in dir
                
            if _pass: #and ("2048" in dir or "4096" in dir):
                models.append((dir,file))
                break
    return models

from pytorch_lightning import Trainer
from CNO_timeModule_CIN import CNO_time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import copy
import json
import os
import sys

import pandas as pd
import torch

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.00075, 
        "weight_decay": 1e-6,
        "scheduler_step": 1,
        "scheduler_gamma": 0.9,
        "epochs": 100,
        "batch_size": 32,         
        "time_steps": 7,          # How many time steps to select?
        "dt": 1,                  # What is the time step? (1 means include entire traj, 2 means taking every other step, etc.
        "training_samples": 32,   # How many training samples?
        "time_input": 1,          # Should we include time in the input channels?
        "allowed": 'all',         # All2ALL (train) - all , or One2All (train) - one2all, AR training - one
        "cluster": True,          # Something internal (don't bother)
    }
    
    model_architecture_ = {
        "N_layers": 4,            # Number of (D) & (U) blocks 
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "N_res": 8,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 8,         # Number of (R) blocks in the BN
        
        "batch_norm": 1,          # Should we use simple BN -- 1: use it? If is_time == 1, we turn it off
        "is_time": 1,             # Should we conditional BN/LN/IN?
        "nl_dim": "23",           # If yes, which norm? '23'-IN, '023'-BN, '123'-LN
        
        "in_size": 128,           # Resolution of the computational grid
        "activation": 'cno_lrelu',# cno_lrelu, cno_lrelu_torch or lrelu
        
        "is_att": False,          # Should we use attention in the bottleneck? You could add it!
        "patch_size" : 1,         # ViT parameters, if is_att == True
        "dim_multiplier" : 1,
        "depth" : 2,
        "heads" : 2,
        "dim_head_multiplier" : 0.5,
        "mlp_dim_multiplier" : 1.0,
        "emb_dropout" : 0.
        }
    
    # AVAILABLE EXPERIMENTS:
    # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
    # "ns_pwc_t:
    # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
    # "rich_mesh", "rayl_tayl" "kolmogorov"
    # "wave_seismic", "wave_gauss", "allen_cahn"
    # "airfoil", "poisson_gauss", "helmholtz"
    
    # FOR PRETRAINING CNO-FM: which_example = "eul_ns_mix1"
    
    # WHAT IS THE EXPERIMENT?
    which_example = "rich_mesh"
    
    folder = "--- PROVIDE THE FOLDER TO SAVE THE MODEL ----" 
    
else:
    raise ValueError("To many args")
    
    

#---------------------------------------------------------    
cluster = True  # We always tun on cluster

model_architecture_["batch_norm"] = True if model_architecture_["batch_norm"] in [1, True] else False
training_properties["time_input"] = True if training_properties["time_input"] in [1, True] else False
model_architecture_["is_att"]     = True if model_architecture_["is_att"]     in [1, True] else False
training_properties["cluster"]    = True if training_properties["cluster"]    in [1, True] else False

if model_architecture_["nl_dim"] in ["023"]:
    model_architecture_["nl_dim"] = [0,2,3]
elif model_architecture_["nl_dim"] in ["123"]:
    model_architecture_["nl_dim"] = [1,2,3]
elif model_architecture_["nl_dim"] in ["23"]:
    model_architecture_["nl_dim"] = [2,3]

#---------------------------------------------------------
if not cluster: # Set the defaulf folder
    folder = "--- PROVIDE THE FOLDER TO SAVE THE MODEL (no cluster) ----"  

if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

#---------------------------------------------------------
# Load parameters related to the specific experiment -- "DataLoaders/all_experiments.json"

Dict_EXP = json.load( open( "DataLoaders/all_experiments.json" ))
if which_example in Dict_EXP:
    loader_dict =  Dict_EXP[which_example]
else:
    raise ValueError("Please specify different benchmark")

# loader_dict: INFORMATION ABOUT THE EXPERIMENT, TRAINING, etc -- VERY IMPORTANT!
loader_dict["which"] = which_example
loader_dict["time_input"] = training_properties["time_input"]
loader_dict["cluster"] = training_properties["cluster"]
loader_dict["num_samples"] = training_properties["training_samples"]
loader_dict["dt"] = training_properties["dt"]
loader_dict["time_steps"] = training_properties["time_steps"]

#---------------------------------------------------------
# Which transitions during the training are allowed?
_allowed = []
if "include_zero" in loader_dict and loader_dict["include_zero"]:
    start_t = 0
else:
    start_t = 1
if training_properties['allowed'] == 'all':
    for t in range(start_t,loader_dict["time_steps"]+1):
        _allowed.append(t)
elif training_properties['allowed']  == "one2all":
    _allowed = None
elif training_properties['allowed'] == 'one':
    _allowed = [1]
loader_dict["allowed_tran"] = _allowed

#---------------------------------------------------------
# Initialize CNO
model   = CNO_time(in_dim =  loader_dict["in_dim"],               
                    in_size  = 128,                  
                    N_layers = model_architecture_["N_layers"],                
                    N_res =  model_architecture_["N_res"], 
                    N_res_neck = model_architecture_["N_res_neck"],           
                    channel_multiplier = model_architecture_["channel_multiplier"],  
                    batch_norm = model_architecture_["batch_norm"],    
                    out_dim = loader_dict["out_dim"],               
                    activation = model_architecture_["activation"], 
                    time_steps = loader_dict["time_steps"],
                    is_time = model_architecture_["is_time"],
                    nl_dim= model_architecture_["nl_dim"],

                    lr = training_properties["learning_rate"],
                    batch_size = training_properties["batch_size"],
                    weight_decay = training_properties["weight_decay"],
                    scheduler_step = training_properties["scheduler_step"],
                    scheduler_gamma = training_properties["scheduler_gamma"],

                    loader_dictionary = loader_dict,

                    is_att = model_architecture_["is_att"],
                    patch_size = model_architecture_["patch_size"],
                    dim_multiplier = model_architecture_["dim_multiplier"],
                    depth = model_architecture_["depth"],
                    heads = model_architecture_["heads"],
                    dim_head_multiplier = model_architecture_["dim_head_multiplier"],
                    mlp_dim_multiplier = model_architecture_["mlp_dim_multiplier"],
                    emb_dropout = model_architecture_["emb_dropout"])

#---------------------------------------------------------

ver = 123 # Just a random string to be added to the model name

checkpoint_callback = ModelCheckpoint(dirpath = folder+"/model"+str(ver), monitor='mean_val_l')
early_part = 10
early_stop_callback = EarlyStopping(monitor="mean_val_l", patience=training_properties["epochs"]//early_part)

logger = TensorBoardLogger(save_dir=folder, version=ver, name="logs")
trainer = Trainer(devices = -1,
                max_epochs = training_properties["epochs"],
                callbacks = [checkpoint_callback,early_stop_callback],
                strategy="ddp_find_unused_parameters_true", #IMPORTANT!!!
                logger=logger)
trainer.fit(model)
trainer.validate(model)

#---------------------------------------------------------
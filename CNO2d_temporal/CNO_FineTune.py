import torch.nn as nn
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from CNO_timeModule_CIN import CNO_time, LiftProjectBlock

import os
import copy
import sys

import itertools
import json

from test_and_fine_tune_utils.fine_tune_utils import load_model, allowed_transitions, initialize_emb

from test_and_fine_tune_utils.fine_tune_lift import initialize_FT

#---------------------------------------------

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        
        properties = {
        "num_trajectories": 128, 
        "epochs": 221,
        
        "lr": 0.00005,        # LR of the base part
        "lr_norm": 0.00125,   # LR of the normalization layers
        "lr_emb": 0.0005,     # LR of the Lift/Project
        "scheduler_step": 5,
        "scheduler_gamma": 0.9,
        
        "is_different_dim": 1, # Are the dimensions different from the pretraining (out-of-context)?
        "in_dim_tune": 4,      # If yes, what are they?
        "out_dim_tune": 3,
        
        "steps": 7             # Time steps in the dataset
        }
        
        # AVAILABLE EXPERIMENTS:
        # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
        # "ns_pwc_t:
        # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
        # "rich_mesh", "rayl_tayl" "kolmogorov"
        # "wave_seismic", "wave_gauss", "allen_cahn"
        # "airfoil", "poisson_gauss", "helmholtz"

        # WHAT IS THE EXPERIMENT?
        which = "kolmogorov"
    
    else:
        
        properties = json.loads(sys.argv[1].replace("\'", "\""))
        which = sys.argv[2]
    
    # What is your Foundation Model?
    folder = "---- PROVIDE THE FOLDER PATH ----"
    in_dim = 5  # in_dim of the foudnarion model
    out_dim = 4 # out_dim of the foudnarion model
    
    #------------------------------------------------------------
    
    # Should we change the lift/project?
    is_different_dim = properties["is_different_dim"] == 1
    in_dim_tune = properties["in_dim_tune"]
    out_dim_tune = properties["out_dim_tune"]
    
    # Num. of trajectories and Num. of steps
    num_trajectories = properties["num_trajectories"]
    steps = properties["steps"]
    
    # Shell we mask (i.e. not predict pressure, etc)?
    if ("ns" in which) and ("_t" not in which):
        is_masked = True #True or None
    else:
        is_masked = None #True or None
    
    # Load the model and change the lift/project if needed
    model, loader_dict = load_model(folder, 
                                    which_example = which,
                                    in_dim = in_dim,
                                    out_dim = out_dim,
                                    steps = steps,
                                    is_masked = is_masked)

    model = initialize_FT(model = model,
                           old_in_dim = 5,
                           new_in_dim = in_dim_tune,
                           new_out_dim = out_dim_tune,
                           old_out_dim = 4)

    #------------------------------------------------------------
    
    batch_size = 32
    epochs = properties["epochs"]
    loader_dict["num_samples"] = num_trajectories
    _allowed = allowed_transitions(one_all = "all", time_steps = steps)
    loader_dict["allowed_tran"] = _allowed
    
    model.scheduler_step = properties["scheduler_step"]
    model.scheduler_gamma = properties["scheduler_gamma"]
    model.loader_dictionary["fine_tuned"] = True
    model.lr     = properties["lr"]
    model.lr_emb = properties["lr_emb"]
    model.lr_norm = properties["lr_norm"]
    model.batch_size = batch_size

    #------------------------------------------------------------
    # CREATE folders
    
    folder_fine_tune = folder+ "/FT_" + which
    if not os.path.isdir(folder_fine_tune):
        print("Generated new folder")
        os.mkdir(folder_fine_tune)
    
    sufix = "traj"+str(num_trajectories)+"_"+"lr_"+ str(properties["lr"])+"_"+str(properties["lr_emb"])+"_"+str(properties["lr_norm"])+"_""epoch"+ str(epochs)
    folder_model = folder_fine_tune +  "/model_fine_tune_" + which + "_" + sufix

    if not os.path.isdir(folder_model):
        print("Generated new folder")
        os.mkdir(folder_model)
    
    #------------------------------------------------------------
    
    print("################")
    print("Foundation Model (FM) ", folder)
    print("EXAMPLE: ", which)
    print("Change L/P ", is_different_dim, in_dim_tune, out_dim_tune)
    print("LR ", properties["lr"])
    print("LR_EBM ", properties["lr_emb"])
    print("LR_NORM ", properties["lr_norm"])
    print("TRAJS ", properties["num_trajectories"])
    print("STEPS ", steps)
    print("################")
    
    #------------------------------------------------------------
    
    ver = 123
    checkpoint_callback = ModelCheckpoint(dirpath = folder_model+"/model"+str(ver), monitor='mean_val_l')

    logger = TensorBoardLogger(save_dir=folder_model, version=ver, name="logs")
    trainer = Trainer(devices = -1,
                    max_epochs = properties["epochs"],
                    callbacks=[checkpoint_callback],
                    strategy="ddp_find_unused_parameters_true",
                    logger=logger)
    trainer.fit(model)
    trainer.validate(model)
    
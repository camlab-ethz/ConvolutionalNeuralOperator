import torch.nn as nn
import torch
from pytorch_lightning import Trainer
from CNO_timeModule_CIN import CNO_time

from torch.utils.data import DataLoader
import os
import copy
import numpy as np

from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model, _load_data, _list_models

import time

#--------------------------------------------------------------------------------------------

def _write_in_file(file_name, 
                   test_dic, 
                   E,
                   E_l1,
                   times, 
                   pattern,
                   separate_var = False,
                   num_var = 0):
    
    file = open(file_name, "w") 

    for key in test_dic:
        #if key in ["dt","max_steps","time_steps","training_samples"]:
        if key == "step_max":
            max_steps = int(float(test_dic[key]))
        elif key == "dt":
            dt = int(float(test_dic[key]))
          
    for t in range(len(times)):
        time = int(times[t]*max_steps)
        
        if not separate_var:
            errors = E[t]
            med_err = np.median(errors)
            mean_err = np.mean(errors)
            file.write("median "+str(time)+","+str(med_err)+"\n")
            file.write("mean " + str(time)+","+str(mean_err)+"\n")
            print(time, med_err)
            
            errors = E_l1[t]
            med_err = np.median(errors)
            mean_err = np.mean(errors)
            file.write("median_l1 "+str(time)+","+str(med_err)+"\n")
            file.write("mean_l1 " + str(time)+","+str(mean_err)+"\n")
            print(time, med_err, "L1 error")
            
        else:
            for i in range(num_var):
                errors_i = E[i,t]
                med_err = np.median(errors_i)
                mean_err = np.mean(errors_i)
                file.write("median "+str(time)+ "_"+ str(i) +","+str(med_err)+"\n")
                file.write("mean "  +str(time)+ "_"+ str(i)+","+str(mean_err)+"\n")
                print(time, i, med_err)
                
                errors_i = E_l1[i,t]
                med_err = np.median(errors_i)
                mean_err = np.mean(errors_i)
                file.write("median_l1 "+str(time)+ "_"+ str(i) +","+str(med_err)+"\n")
                file.write("mean_l1 "  +str(time)+ "_"+ str(i)+","+str(mean_err)+"\n")
                print(time, i, med_err, "L1 error")
        
    file.close()
    
def _test_pattern(model, 
                testing_loader,
                batch_size = 10,
                time_steps = 5,
                dt = 2,
                max_steps = 10.0,
                pattern = [1,1,1,1,1],
                device = 'cuda',
                folder = "",
                separate_var = False,
                separate_dim = [],
                is_masked = None,
                steady    = False,
                is_airfoil = False):
    
    print("batch_size", batch_size)
    
    delta = dt/max_steps
    T = []    
    E = np.zeros((len(pattern), 0))
    E_l1 = np.zeros((len(pattern), 0))
    
    std_inp = 1
    std_out = 1
    
    if separate_var:
        len_sep = len(separate_dim)
        diff = [0, separate_dim[0]]
        for i in range(1,len_sep):
            diff.append(diff[-1]+separate_dim[i])
        E = np.zeros((len_sep,len(pattern), 0))
        E_l1 = np.zeros((len_sep,len(pattern), 0))
    
    #-----------------
    
    print(len(testing_loader), "testing_loader")
    
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(testing_loader):
            
            if is_masked:
                time_batch_, input_batch_, output_batch_, masked_dim = batch
            else:
                time_batch_, input_batch_, output_batch_ = batch
            
            acc_jump = 0

            t_batch = copy.deepcopy(time_batch_.to(device))
            input_batch = copy.deepcopy(input_batch_.to(device))
            output_batch = copy.deepcopy(output_batch_.to(device))
            dim = output_batch.shape[1]
                        
            for i,jump in enumerate(pattern):
                acc_jump+=jump

                if step == 0:
                    T.append(acc_jump*delta)
                
                if i == 0:
                    inp  = input_batch[0::time_steps]
                    zero_time_inp = torch.zeros_like(t_batch[0::time_steps]).to(device)
                    
                    if not separate_dim:
                        E_curr = np.zeros((len(pattern), inp.shape[0]))
                        E_curr_l1 = np.zeros((len(pattern), inp.shape[0]))
                    else:
                        E_curr = np.zeros((len_sep, len(pattern), inp.shape[0]))
                        E_curr_l1 = np.zeros((len_sep, len(pattern), inp.shape[0]))
                        
                time_batch = zero_time_inp + jump * delta   
                
                if not steady:
                    inp[:,-1] = jump * delta
                else:
                    time_batch = zero_time_inp + 1.0

                out  = copy.deepcopy(output_batch_[acc_jump - 1::time_steps]).to(device)
                pred = model(inp, time_batch)

                    
                if is_airfoil:
                    out[input_batch==1] = 1.0
                    pred[input_batch==1] = 1.0
                
                if not separate_dim:
                    err = (torch.mean(abs(pred - out), (-3, -2, -1)) / torch.mean(abs(out), (-3, -2, -1)))* 100
                    err = err.detach().cpu().numpy()
                    E_curr[i] = err
                    
                    err = torch.mean(abs(pred - out), (-3, -2, -1))
                    err = err.detach().cpu().numpy()
                    E_curr_l1[i] = err
                
                else:
                    for j in range(len_sep):
                        
                        
                        if is_masked:
                            mask = masked_dim[0,diff[j]:diff[j+1]]
                            mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(pred.shape[0], mask.shape[0], model.encoder_sizes[0], model.encoder_sizes[0])
                            pred[:,diff[j]:diff[j+1]][mask==0.0] = 1.0
                            out[:,diff[j]:diff[j+1]][mask==0.0] = 1.0
                        
                        
                        err = (torch.mean(abs(pred[:,diff[j]:diff[j+1]] - out[:,diff[j]:diff[j+1]]), (-3, -2, -1)) / (torch.mean(abs(out[:,diff[j]:diff[j+1]]), (-3, -2, -1)) + 1e-10))* 100
                        err = err.detach().cpu().numpy()
                        E_curr[j,i] = err
                        
                        err = torch.mean(abs(pred[:,diff[j]:diff[j+1]] - out[:,diff[j]:diff[j+1]]), (-3, -2, -1))
                        err = err.detach().cpu().numpy()
                        E_curr_l1[j,i] = err
                        
                        if is_masked:
                            # Reinitialize for AR usage (pressure must be zero)
                            pred[:,diff[j]:diff[j+1]][mask==0.0] = 0.0
                        
                inp[:,:dim]  = pred[:,:dim] * (std_out/std_inp)                    
            
            E = np.concatenate((E, E_curr), axis = -1)
            E_l1 = np.concatenate((E_l1, E_curr_l1), axis = -1)
            
            
            if step%10==0:
                print(step,jump, len(testing_loader))
                
            
    return T, E, E_l1

# If pattern = [1, 2, 2] you predict recursively:
#  t(0) -> t(1) -> t(3) -> t(5)
def _evaulate_models(folders,
                    which_example = "shear_layer",
                    cluster = False,
                    in_dist = True,
                    patterns = [[1,1,1,1,1]],
                    folder_pattern = "errors",
                    additional_flags = [],
                    
                    steps = 10,
                    steady = False,
                    
                    separate_var = False,
                    is_masked = False,
                    fine_tuned = False,
                    training_folder = None,
                    write_in_file = True,
                     
                    old_in_dim  = 5,
                    new_in_dim  = 5,
                    new_out_dim = 4):
    
    model_dirs = _list_models(folders, additional_flags)
    
        
    if len(model_dirs) == 0:
        return None
    
    is_loaded = False

    if in_dist:
        in_dist_sufix = "in_dist_patterns.txt"
    else:
        in_dist_sufix = "out_dist_patterns.txt"
    
    #print(model_dirs)
    #return None
    
    #-----------------------------------------
    
    # ('Scaling_NS/Light_ShearLayer_Scaling_l5t2_is0bn1_1/2500Setup_2', 'epoch=1919-step=40320.ckpt')
    for folder, _model_file in model_dirs:
        
        print(folder, _model_file)
        _subfolders = os.listdir(folder)  
        _pass = False #Should we do anything?
        

        if folder_pattern not in _subfolders:
            _pass = True
        
        if _pass:
                        
            if not fine_tuned and training_folder is None:
                _folder = folder
            else:
                _folder = training_folder
            
            train_file  = _folder + "/training_properties.txt"
            net_file    = _folder + "/net_architecture.txt"
            
            loader_dict = _load_dict(files = [train_file, net_file], 
                                     which_example = which_example, 
                                     steps = steps,
                                     is_masked = is_masked)
            
            print(loader_dict)
            bs = 32*7
            
            if bs%steps != 0:
                bs = (bs//steps)*steps
            
            if not is_loaded:
                test_data_loader  = _load_data(which = which_example, 
                                            loader_dict = loader_dict,
                                            batch_size = bs,
                                            fix_input_to_time_step = 0)
            print(len(test_data_loader), "test_data_loader", bs)
            #########################
            # Model init. and loading
            #########################
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_file = folder + "/model123/" + _model_file
            
            print(loader_dict["in_dim"], loader_dict["out_dim"], "DIMENSIONS")
            model = _initialize_model(loader_dict,
                                      diff_embedding = fine_tuned,
                                      old_in_dim  = old_in_dim,
                                      new_in_dim  = loader_dict["in_dim"],
                                      new_out_dim = loader_dict["out_dim"])
            
            checkpoint = torch.load(model_file, map_location = device)
            model.load_state_dict(checkpoint["state_dict"])

            print("DT",loader_dict["dt"])
            model = model.to(device)
                        
            if separate_var:
                separate_dim = loader_dict["separate_dim"]
            else:
                separate_dim = []
            
            #########
            
            to_write = False
            
            if not os.path.exists(folder+"/"+folder_pattern):
                to_write = True
                os.mkdir(folder+"/"+folder_pattern)
            
            for pattern in patterns: 

                if which_example == "airfoil":
                    is_airfoil = True
                else:
                    is_airfoil = False
                T, E, E_l1 = _test_pattern(model = model, 
                                        testing_loader = test_data_loader,
                                        dt = loader_dict["dt"],
                                        time_steps = loader_dict["time_steps"],
                                        max_steps = loader_dict["step_max"],
                                        pattern = pattern,
                                        device = device,
                                        batch_size = loader_dict["time_steps"],
                                        folder = folder+"/"+folder_pattern,
                                        separate_var = separate_var,
                                        separate_dim = separate_dim,
                                        is_masked = is_masked,
                                        steady = steady,
                                        is_airfoil = is_airfoil)
                
                
                #########################
                # What is the error file?
                #########################
                
                s = ""
                for p in pattern:
                    s = s + str(p) + "_"

                error_file = folder+"/"+folder_pattern+"/"+ s + in_dist_sufix
                
                #################
                # Write in file:
                #################
                
                if write_in_file and to_write:
                    
                    _write_in_file(error_file, 
                                   loader_dict, 
                                   E,
                                   E_l1,
                                   T, 
                                   pattern,
                                   separate_var = separate_var,
                                   num_var = len(separate_dim))
            
            print("----------------------------------")

#-----------------------------------------------------------------

if __name__ == "__main__":

    
    # AVAILABLE EXPERIMENTS:
    # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
    # "ns_pwc_t:
    # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
    # "rich_mesh", "rayl_tayl" "kolmogorov"
    # "wave_seismic", "wave_gauss", "allen_cahn"
    # "airfoil", "poisson_gauss", "helmholtz"
    
    # WHAT IS THE EXPERIMENT?
    which_example = "kolmogorov"
    
    # IS THE MODEL FINETUNED?
    fine_tuned = False
    old_in_dim  = 5 #PRETRAINING INPUT DIMENSION -- RELEVANT FOR FINETUNNING
    
    if not fine_tuned:
        
        # PROVIDE THE FOLDER WITH ALL THE MODELS FOR SCALING LAW::
        folders = "---- PROVIDE THE FOLDER PATH ----"
        training_folder = None # Keep it None
    
    else:
        # FOLDER OF THE FINETUNED MODEL:
        folders = "---- PROVIDE THE FOLDER PATH ----" 
        
        # FOLDER WHERE THE TRAINING DETAILS ARE (training_properties.txt file):
        training_folder = "---- PROVIDE THE FOLDER PATH ----"
    
    #------------------
    # Other parameters:
    #------------------
    
    
    folder_pattern = "errors" # WHAT IS THE NAME OF THE FOLDER FOR SAVING ERRORS?
    steps   = 7               # HOW MANY STEPS ARE THERE IN THE TRAJECTORY?
    additional_flags =  []    # ANY ADDITIONAL CONSTRAINTS IN THE NAMES OF THE FILES?
    is_masked = False         # ARE I/O MASKED?
    write_in_file = True      # SHOULD WE WRITE THE ERRORS IN THE FILE?
    
    # ARE THE VARIABLES SEPARATED? (See test_and_fine_tune_utils.test_utils._load_dict for details)
    separate_var = True 
    
    # IS THE EPXERIMENT STEADY?
    steady  =  "poisson" in which_example or "helmholtz" in which_example or "airfoil" in which_example 
    
    # DIFFERENT EVALUATION SCHEMES (DIRECT, HETEROG. AR, HOMOG. AR)
    if not steady:
        patterns = [[7], [2,2,2,1], [1, 1, 1, 1, 1, 1, 1]]
    else:
        patterns = [[1]]
    
    #----------
    # EVALUATE:
    #----------
    _evaulate_models(folders = folders,
                    which_example = which_example,
                    cluster = True,
                    in_dist = True,
                    patterns = patterns,
                    folder_pattern = folder_pattern,
                    steps = steps,
                    additional_flags = additional_flags,
                    steady = steady,
                    separate_var = separate_var,
                    is_masked = is_masked,
                    training_folder = training_folder,
                    write_in_file = write_in_file,
                    fine_tuned = fine_tuned,
                    old_in_dim  = old_in_dim)


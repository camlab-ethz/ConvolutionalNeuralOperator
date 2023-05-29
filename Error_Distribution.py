import random

import h5py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
#import seaborn as sn
import torch
import json
import ast
import os
import scipy
import torch.nn as nn

#import seaborn as sns
import pandas as pd
from scipy import stats

import scipy
from random import randint

#------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(folder, which_model, device, which_example, in_size = 64, batch_size = 32, training_samples = 1, in_dist = True):
    
    if which_model == "CNO":
        from Problems.Benchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
    elif which_model == "FNO":
        from Problems.FNOBenchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer

    model_architecture_ = dict()
    with open(folder + "/net_architecture.txt") as f:
        for line in f:
            key_values = line.replace("\n", "").split(",")
            key = key_values[0]
            value = ast.literal_eval(key_values[1])
            model_architecture_[key] = int(value)
    model_architecture_["in_size"] = in_size
    
    if which_model == "CNO" or which_model == "UNET":
        if which_example == "shear_layer":
            example = ShearLayer(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)
        elif which_example == "poisson":
            example = SinFrequency(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "wave_0_5":
            example = WaveEquation(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "allen":
            example = AllenCahn(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "cont_tran":
            example = ContTranslation(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "disc_tran":
            example = DiscContTranslation(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)
        elif which_example == "airfoil":
            example = Airfoil(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)

        else:
            raise ValueError()
    
    elif which_model == "FNO":
        
        if which_example == "shear_layer":
            example = ShearLayer(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)
        elif which_example == "poisson":
            example = SinFrequency(model_architecture_, device, batch_size, s = 64, in_dist = in_dist)
        elif which_example == "wave_0_5":
            example = WaveEquation(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "allen":
            example = AllenCahn(model_architecture_, device, batch_size, in_dist = in_dist)
        elif which_example == "cont_tran":
            example = ContTranslation(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)
        elif which_example == "disc_tran":
            example = DiscContTranslation(model_architecture_, device, batch_size,training_samples, in_dist = in_dist)
        elif which_example == "airfoil":
            example = Airfoil(model_architecture_, device, batch_size, training_samples, in_dist = in_dist)
        else:
            raise ValueError()
    
    testing_loader = example.test_loader
    
    return testing_loader

#------------------------------------------------------------------------------------------------------------------------------------------------

def error_distribution(which_model, model, testing_loader, p, N, device, which = "shear_layer"):
    E_diss = np.zeros(N)
    cnt = 0
    
    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(testing_loader):
            
            batch = inputs.shape[0]            

            if i>=N:
                break
                
            if which_model == "CNO" or which_model == "UNET":
                
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                prediction = model(inputs)

                if which == "airfoil":
                    outputs[inputs==1] = 1
                    prediction[inputs==1] = 1
                
                err = (torch.mean(abs(prediction[:,0,:,:] - outputs[:,0,:,:]) ** p, (-2, -1)) / torch.mean(abs(outputs[:,0,:,:]) ** p, (-2, -1))) ** (1 / p) * 100
                
            elif which_model == "FNO":
                inputs = inputs.to(model.device)
                outputs = outputs.to(model.device)
                prediction = model(inputs)
                
                if which == "airfoil":
                    outputs[inputs==1] = 1
                    prediction[inputs==1] = 1
                
                err = (torch.mean(abs(prediction[:,:, :, 0] - outputs[:, :, :, 0]) ** p, (-2, -1)) / torch.mean(abs(outputs[:, :, :,0]) ** p, (-2, -1))) ** (1 / p) * 100
            
            else:
                return None
            
            E_diss[cnt: cnt + batch] = err.detach().cpu().numpy()
            cnt+=batch
    
            
    
    return E_diss

def write_in_file_distribution(folder, file, E):
    
    hfile = h5py.File(folder+"/"+file, "w")
    hfile.create_group("dist")
    hfile["dist"].create_dataset("E", data=E)
    hfile.close()

#------------------------------------------------------------------------------------------------------------------------------------------------

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

    
def avg_spectra(which, model, data_loader, device, N = 128, in_size = 64):
    
    avg = np.zeros((in_size, in_size))
    avg_inp = np.zeros((in_size, in_size))
    avg_out = np.zeros((in_size, in_size))
    
    with torch.no_grad():
        
        for i, (inputs, outputs) in enumerate(data_loader):
        
            inputs = inputs.to(device)
            prediction = model(inputs)

            if which == "FNO":
                prediction = prediction.permute(0,3,1,2)
                inputs     = inputs.permute(0,3,1,2)
                outputs    = outputs.permute(0,3,1,2)
                
            prediction = prediction.detach().cpu().numpy()[:,0,:,:]
            inputs     = inputs.detach().cpu().numpy()[:,0,:,:]
            outputs    = outputs.detach().cpu().numpy()[:,0,:,:]
            
            avg = avg +  np.sum(np.absolute(samples_fft(prediction)), 0)
            avg_inp = avg_inp +  np.sum(np.absolute(samples_fft(inputs)), 0)
            avg_out = avg_out +  np.sum(np.absolute(samples_fft(outputs)), 0)
            
        
    return avg/N, avg_inp/N, avg_out/N

def write_in_file_spectra(folder, file, avg, avg_inp, avg_out):
    
    hfile = h5py.File(folder+"/"+file, "w")
    hfile.create_group("spectra")
    hfile["spectra"].create_dataset("avg", data=avg)
    hfile["spectra"].create_dataset("avg_inp", data=avg_inp)
    hfile["spectra"].create_dataset("avg_out", data=avg_out)
    hfile.close()


#------------------------------------------------------------------------------------------------------------------------------------------------

def plot_samples(which_model, data_loader, model, p, n, cmap = "jet", which = "shear_layer"):

    cmap = "gist_ncar"

    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(data_loader):
            if i == n:
                    
                inputs = inputs.to(model.device)
                outputs = outputs.to(model.device)
                prediction = model(inputs)
                
                if which == "airfoil":
                    outputs[inputs==1] = 2
                    prediction[inputs==1] = 2
                    inputs[inputs==1] = 2 
                    
                fig, axes = plt.subplots(1, 3, figsize=(26, 6))
                axes[0].grid(True, which="both", ls=":")
                axes[1].grid(True, which="both", ls=":")
                axes[2].grid(True, which="both", ls=":")
                fontsize = 15
                
                vmax = 1
                vmin = 0
                if which == "airfoil":
                    vmax = 1.25
                    vmin = 0.5
                                
                if which_model == "CNO" or which_model == "UNET":
                    
                    axes[0].invert_yaxis()
                    im1 = axes[0].imshow(inputs[0,0,:, :].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin, extent=(0,1,0,1))
                    axes[0].title.set_text("Input " + which_model)
                    
                    axes[1].invert_yaxis()
                    im2 = axes[1].imshow((prediction[0,0,:, :]).detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    axes[1].title.set_text("Prediction "+ which_model)
                    
                            
                    axes[2].invert_yaxis()
                    im3 = axes[2].imshow(outputs[0,0,:, :].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    plt.title("Ground Truth")
                    
                elif which_model == "FNO":

                    axes[0].invert_yaxis()
                    im1 = axes[0].imshow(inputs[0,:, :,0].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin, extent=(0,1,0,1))
                    axes[0].title.set_text("Input " + which_model)
                    
                    axes[1].invert_yaxis()
                    im2 = axes[1].imshow(prediction[0,:, :,0].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    axes[1].title.set_text("Prediction "+ which_model)
                    
                
                    axes[2].invert_yaxis()
                    im3 = axes[2].imshow(outputs[0,:, :,0].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    plt.title("Ground Truth")
                    
                    
                    # -------------------------------------------------------------
                
                axes[0].xaxis.set_tick_params(labelsize=fontsize)
                axes[0].yaxis.set_tick_params(labelsize= fontsize)
                axes[1].xaxis.set_tick_params(labelsize=fontsize)
                axes[1].yaxis.set_tick_params(labelsize= fontsize)
                axes[2].xaxis.set_tick_params(labelsize=fontsize)
                axes[2].yaxis.set_tick_params(labelsize= fontsize)
                
                axes[0].title.set_size(fontsize+6)
                axes[1].title.set_size(fontsize+6)
                axes[2].title.set_size(fontsize+6)
                    
                fig.colorbar(im3)
                plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------


def plot_samples_different_models(which_models, data_loaders, models, p, n, which = "shear_layer"):
   
    preds = []
    
    for w, data_loader in enumerate(data_loaders):
        
        which_model = which_models[w]
        model = models[w]
        
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(data_loader):
            
                #print(i, n)
                #print(which_model, "Which")

                if i == n:

                    inputs = inputs.to(model.device)
                    outputs = outputs.to(model.device)
                    
                    prediction = model(inputs)
                
                
                    if which == "airfoil":
                        outputs[inputs==1] = 1
                        prediction[inputs==1] = 1
                        inputs[inputs==1] = 1
                
                    if which_model == "CNO"  or which_model == "UNET":
                        preds.append(prediction[0,0,:, :].detach().numpy())
                        
                        if w == 0:
                            inp = inputs[0,0,:, :].detach().numpy()
                            out = outputs[0,0,:, :].detach().numpy()
                        
                    elif which_model == "FNO":
                        preds.append(prediction[0,:, :,0].detach().numpy())

                        
                        if w == 0:
                            inp = inputs[0,:, :,0].detach().numpy()
                            out = outputs[0,:, :,0].detach().numpy()
    
    M = len(which_models)
    fig, axes = plt.subplots(1, M+2, figsize=(5*(M+2)+3, 4))
    #fig, axes = plt.subplots(1, M+2)

    for m in range(M+2):
        axes[m].grid(True, which="both", ls=":")
   
    fontsize = 16
    
    
    if which == "airfoil":
        inp[inp==1] = np.max(out)
        out[out==1] = np.max(out)
    
    cmap = "gist_ncar" 
    
    im1 = axes[0].imshow(inp, cmap=cmap,extent = (0,1,1,0),  interpolation = "bicubic")
    axes[0].invert_yaxis()

    axes[0].title.set_text("Input")
    
    im2 = axes[1].imshow(out, cmap=cmap,extent = (0,1,1,0),interpolation = "bicubic")
    axes[1].title.set_text("Output")
    axes[1].invert_yaxis()

    for m in range(M):
        preds[m][preds[m]==1] = np.max(out)
        im3 = axes[m+2].imshow(preds[m], cmap=cmap,extent=(0,1,1,0),interpolation = "bicubic")
        axes[m+2].title.set_text("Prediction "+ which_models[m])
        axes[m+2].invert_yaxis()

    for m in range(M+2):
        axes[m].xaxis.set_tick_params(labelsize=fontsize)
        axes[m].yaxis.set_tick_params(labelsize= fontsize)
        axes[m].title.set_size(fontsize+10)
    
    plt.colorbar(im3)
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------

# Set your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   "which" can be 

#   poisson             : Poisson equation 
#   wave_0_5            : Wave equation
#   cont_tran           : Smooth Transport
#   disc_tran           : Discontinuous Transport
#   allen               : Allen-Cahn equation
#   shear_layer         : Navier-Stokes equations
#   airfoil             : Compressible Euler equations

which = "allen"
in_dist = False

#Do you want to plot or to compute MEDIAN L1 errors? 
plot  = False 

if plot:
    device = "cpu"
    dist = False
else:
    dist = True

# Model folders
if which == "poisson":
    folder_UNET = "SELECTED_MODELS/Best_poisson_UNet"
    folder_CNO = "SELECTED_MODELS/Best_poisson_CNO"
    folder_FNO = "SELECTED_MODELS/Best_poisson_FNO"

    
    N = 256

elif which == "wave_0_5":
    folder_UNET = "SELECTED_MODELS/Best_wave_UNet"
    folder_CNO = "SELECTED_MODELS/Best_wave_CNO" 
    folder_FNO = "SELECTED_MODELS/Best_wave_FNO" 

    N = 256

elif which == "allen":
    folder_UNET = "SELECTED_MODELS/Best_allen_cahn_UNet" 
    folder_CNO = "SELECTED_MODELS/Best_allen_cahn_CNO" 
    folder_FNO = "SELECTED_MODELS/Best_allen_cahn_FNO" 

    N = 128
    
elif which == "shear_layer":
    folder_UNET = "SELECTED_MODELS/Best_shear_layer_UNet"
    folder_CNO = "SELECTED_MODELS/Best_shear_layer_CNO"
    folder_FNO = "SELECTED_MODELS/Best_shear_layer_FNO"

    N = 128


elif which == "cont_tran":
    folder_UNET = "SELECTED_MODELS/Best_cont_t_UNet"
    folder_CNO = "SELECTED_MODELS/Best_cont_t_CNO"
    folder_FNO = "SELECTED_MODELS/Best_cont_t_FNO"

    N = 256

elif which == "disc_tran":
    folder_UNET = "SELECTED_MODELS/Best_discont_t_UNet" 
    folder_CNO = "SELECTED_MODELS/Best_discont_t_CNO" 
    folder_FNO = "SELECTED_MODELS/Best_discont_t_FNO" 

    N = 256

elif which == "airfoil":
    folder_UNET = "SELECTED_MODELS/Best_airfoil_UNet"  
    folder_CNO = "SELECTED_MODELS/Best_airfoil_CNO"  
    folder_FNO = "SELECTED_MODELS/Best_airfoil_FNO"  

    N = 128

if dist:
    modelFNO = torch.load(folder_FNO + "/model.pkl", map_location=torch.device(device))
    modelCNO = torch.load(folder_CNO + "/model.pkl", map_location=torch.device(device))
    modelUNET = torch.load(folder_UNET + "/model.pkl", map_location=torch.device(device)) 
    
    modelFNO.device = device
    modelCNO.device = device
    modelUNET.device = device
    
    data_loader_FNO = load_data(folder_FNO, "FNO", device, which, in_dist = in_dist)
    data_loader_CNO = load_data(folder_CNO, "CNO", device, which, in_dist = in_dist)
    data_loader_UNET = data_loader_CNO
    
    data_loader_FNO.num_workers = 16
    data_loader_CNO.num_workers = 16
    data_loader_UNET.num_workers = 16
    
    E_FNO = error_distribution(which_model = "FNO", model = modelFNO, testing_loader = data_loader_FNO, p = 1, N = N, device = device, which = which)
    E_CNO = error_distribution(which_model = "CNO", model = modelCNO, testing_loader = data_loader_CNO, p = 1, N = N, device = device, which = which)
    E_UNET = error_distribution(which_model = "UNET", model = modelUNET, testing_loader = data_loader_CNO, p = 1, N = N, device = device, which = which)    
    
    if which == "airfoil":
        size = 128
    else:
        size = 64
    
    #Write distributions?
    write_dist = False
    if write_dist:
        dist_name = "dist.h5"
        write_in_file_distribution(folder_CNO, dist_name, E_CNO)
        write_in_file_distribution(folder_FNO, dist_name, E_FNO)
        write_in_file_distribution(folder_UNET, dist_name, E_UNET)
    
    #Write avg. spectra?
    write_spectra = False
    if write_spectra:
        spectra_name = "Spectra.h5"
        
        avg, avg_inp, avg_out = avg_spectra("CNO", modelCNO, data_loader_CNO, device, in_size = size)
        write_in_file_spectra(folder_CNO, spectra_name, avg, avg_inp, avg_out)
        
        avg, avg_inp, avg_out = avg_spectra("CNN", modelUNET, data_loader_UNET, device, in_size = size)
        write_in_file_spectra(folder_UNET, spectra_name, avg, avg_inp, avg_out)
        
        avg, avg_inp, avg_out = avg_spectra("FNO", modelFNO, data_loader_FNO, device, in_size = size)
        write_in_file_spectra(folder_FNO,spectra_name, avg, avg_inp, avg_out)
        
        dist_name = "dist.h5"
        write_in_file_distribution(folder_CNO, dist_name, E_CNO)
        write_in_file_distribution(folder_FNO, dist_name, E_FNO)
        write_in_file_distribution(folder_UNET, dist_name, E_UNET)
    
    print("-------------")
    print("Experiment: ", which)
    print("in_dist = " + str(in_dist))
    print("")
    print("CNO error:", np.median(E_CNO))
    print("FNO error:", np.median(E_FNO))
    print("UNet error:", np.median(E_UNET))
    print("-------------")


elif plot:
        
    modelFNO = torch.load(folder_FNO + "/model.pkl", map_location=torch.device(device))
    modelCNO = torch.load(folder_CNO + "/model.pkl", map_location=torch.device(device))
    modelUNET = torch.load(folder_UNET + "/model.pkl", map_location=torch.device(device)) 
    
    modelFNO.device = device
    modelCNO.device = device
    modelUNET.device = device
    
    in_dist = False
    
    data_loader_FNO = load_data(folder_FNO, "FNO", device, which, batch_size = 1, in_dist = in_dist)
    data_loader_CNO = load_data(folder_CNO, "CNO", device, which, batch_size = 1, in_dist = in_dist)
    data_loader_UNET = data_loader_CNO
    
    random.seed()
    n = randint(0,N)

    data_loader_FNO.num_workers = 0
    data_loader_CNO.num_workers = 0

    random.seed()
    n = randint(0,N)
    plot_samples_different_models(["CNO", "FNO", "UNET"], 
                                  [data_loader_CNO, data_loader_FNO, data_loader_UNET], 
                                  [modelCNO, modelFNO, modelUNET], 
                                  1, n, which = which)
    

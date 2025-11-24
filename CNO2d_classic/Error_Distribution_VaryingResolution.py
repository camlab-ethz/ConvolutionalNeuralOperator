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


def load_data(folder, which_model, device, which_example, in_size = 64, batch_size = 32, training_samples = 1, s = 64):
    
    if which_model == "CNO":
        from Problems.CNOBenchmarks import ShearLayer
    elif which_model == "FNO":
        from Problems.FNOBenchmarks import ShearLayer
        
    model_architecture_ = dict()
    with open(folder + "/net_architecture.txt") as f:
        for line in f:
            key_values = line.replace("\n", "").split(",")
            key = key_values[0]
            value = ast.literal_eval(key_values[1])
            model_architecture_[key] = int(value)
    model_architecture_["in_size"] = in_size
    
    if which_model == "CNO":
        if which_example == "shear_layer":
            model_architecture_["in_size"] = s
            example = ShearLayer(model_architecture_, device, batch_size, training_samples)
        else:
            raise ValueError()
    elif which_model == "FNO":
        if which_example == "shear_layer":
            model_architecture_["in_size"] = s
            example = ShearLayer(model_architecture_, device, batch_size, training_samples)
        else:
            raise ValueError()
    else:
        raise ValueError()
    
    testing_loader = example.test_loader
    
    return testing_loader

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def samples_fft(u):
    return scipy.fft.fftn(u, s=u.shape[2:], norm='forward', workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifftn(u_hat, s=u_hat.shape[2:], norm='forward', workers=-1).real


def downsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    u_hat_down = None
    if d == 2:
        u_hat_down = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_down[:,:,:N//2+1,:N//2+1] = u_hat[:,:,:N//2+1,:N//2+1]
        u_hat_down[:,:,:N//2+1,-N//2:] = u_hat[:,:,:N//2+1,-N//2:]
        u_hat_down[:,:,-N//2:,:N//2+1] = u_hat[:,:,-N//2:,:N//2+1]
        u_hat_down[:,:,-N//2:,-N//2:] = u_hat[:,:,-N//2:,-N//2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_down
    u_down = samples_ifft(u_hat_down)
    return u_down


def upsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f'Expected either real or complex valued array. Got {u.dtype}.')
    d = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    u_hat_up = None
    if d == 2:
        u_hat_up = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_up[:,:,:N_old//2+1,:N_old//2+1] = u_hat[:,:,:N_old//2+1,:N_old//2+1]
        u_hat_up[:,:,:N_old//2+1,-N_old//2:] = u_hat[:,:,:N_old//2+1,-N_old//2:]
        u_hat_up[:,:,-N_old//2:,:N_old//2+1] = u_hat[:,:,-N_old//2:,:N_old//2+1]
        u_hat_up[:,:,-N_old//2:,-N_old//2:] = u_hat[:,:,-N_old//2:,-N_old//2:]
    else:
        raise ValueError(f'Invalid dimension {d}')
    if fourier:
        return u_hat_up
    u_up = samples_ifft(u_hat_up)
    return u_up

#FNO --> 

def resize(x, out_size, permute=False):
    if permute:
        x = x.permute(0, 3, 1, 2)
        
    f = torch.fft.rfft2(x, norm='backward')
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1]//2 + 1), dtype=f.dtype, device=f.device)
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    # x_z = torch.fft.ifft2(f_z, s=out_size).real
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])
 
    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]
 
    if permute:
        x_z = x_z.permute(0, 2, 3, 1)
    return x_z

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def error_distribution(which_model, model, testing_loader, p, N, device, s = 64, s_train = 64, which = "shear_layer"):
    E_diss = np.zeros(N)
    cnt = 0
    
    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(testing_loader):
            
            batch = inputs.shape[0]            
                        
            if i>=N:
                break
            
            if i%50==0:
                print(i, N)
                
            if which_model == "CNN":
                
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                prediction = model(inputs)
                
                if which == "airfoil":
                    outputs[inputs==1] = 1
                    prediction[inputs==1] = 1
                
                err = (torch.mean(abs(prediction[:,0,:,:] - outputs[:,0,:,:]) ** p, (-2, -1)) / torch.mean(abs(outputs[:,0,:,:]) ** p, (-2, -1))) ** (1 / p) * 100
                
            elif which_model == "CNO":
                
                inputs = inputs.to(device)
                
                if s!=s_train:
                    inputs = resize(inputs, (s_train, s_train))
                    prediction = model(inputs)
                    prediction = resize(prediction, (s, s))
                else:
                    prediction = model(inputs)
                
                outputs = outputs.to(device)
                
                if which == "airfoil":
                    outputs[inputs==1] = 1
                    prediction[inputs==1] = 1
                
                err = (torch.mean(abs(prediction[:,0,:,:] - outputs[:,0,:,:]) ** p, (-2, -1)) / torch.mean(abs(outputs[:,0,:,:]) ** p, (-2, -1))) ** (1 / p) * 100
            
            else:
                inputs = inputs.to(model.device)
                outputs = outputs.to(model.device)        
                prediction = model(inputs)
                
                if which == "airfoil":
                    outputs[inputs==1] = 1
                    prediction[inputs==1] = 1
                
                err = (torch.mean(abs(prediction[:,:, :, 0] - outputs[:, :, :, 0]) ** p, (-2, -1)) / torch.mean(abs(outputs[:, :, :,0]) ** p, (-2, -1))) ** (1 / p) * 100

            E_diss[cnt: cnt + batch] = err.detach().cpu().numpy()
            cnt+=batch
    
    return E_diss

#-------------------------------------------------------------------------------------------

def plot_samples(which_model, data_loader, model, p, n, s, s_train, cmap = "jet", which = "shear_layer"):

    cmap = "gist_ncar"
    
    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(data_loader):
            
            if i == n:
                    
                print(i, n)
                    
                inputs = inputs.to(model.device)
                outputs = outputs.to(model.device)
                
                if which_model == "CNO":
                    if s!=s_train:
                        inputs1 = resize(inputs, (s_train, s_train))
                        prediction = model(inputs1)
                        prediction = resize(prediction, (s, s))
                        
                    else:
                        prediction = model(inputs)

                else:
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
                                
                if which_model == "CNO" or which_model == "CNN":
                    print("SJDSD")
                    
                    axes[0].invert_yaxis()
                    im1 = axes[0].imshow(inputs[0,0,:, :].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin, extent=(0,1,0,1))
                    axes[0].title.set_text("Input " + which_model)
                    
                    axes[1].invert_yaxis()
                    im2 = axes[1].imshow((prediction[0,0,:, :]).detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    axes[1].title.set_text("Prediction "+ which_model)
                    
                            
                    axes[2].invert_yaxis()
                    im3 = axes[2].imshow(outputs[0,0,:, :].detach().numpy().T, cmap=cmap, vmax = vmax, vmin = vmin,extent=(0,1,0,1))
                    plt.title("Ground Truth")
                    
                    err = (torch.mean(abs(prediction[:,0,:,:] - outputs[:,0,:,:]) ** p, (-2, -1)) / torch.mean(abs(outputs[:,0,:,:]) ** p, (-2, -1))) ** (1 / p) * 100
                    print("CNN", err.item())
                    
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

#-------------------------------------------------------------------------------------------

def plot_different_resolution(D, YTicks, title, loc_legend = "upper right"):
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    axes.grid(True, which="both", ls=":")
    fontsize = 20
    linewidth = 3.5
    
    first = True
    
    for k in D:
        E, _min, _max, _min_res, _index_t, _ds, _t_res, color = D[k]
        R = np.arange(_min_res, _max + 1, _ds)
        
        axes.plot(R, E, "-o", zorder = 3, label = k, color = color, linewidth = linewidth)
        
        if first:
            axes.scatter([_t_res], [E[_index_t]], color = 'red', label = "Training res.", marker = "o", zorder=5, s = 75)
            first = False
        else:
            axes.scatter([_t_res], [E[_index_t]], color = 'red', marker = "o", zorder=5, s = 75)
        
        if _min_res != _min:
            axes.scatter([_min_res], E[0], color = 'orange', label = "Minimal res. ", marker = "o",  zorder = 100, s = 75)

    if _t_res == 64:
        XTicks = [32,64, 96, 128]
    else:
        raise ValueError("Not a good _t_res")
    axes.set_xticks(XTicks)
    axes.set_yticks(YTicks)
    axes.legend(fontsize = fontsize, loc = loc_legend)
    axes.xaxis.set_tick_params(labelsize=fontsize+10)
    axes.yaxis.set_tick_params(labelsize= fontsize+10)
    axes.set_xlabel("Testing Resolution", fontsize = fontsize+10)
    axes.set_ylabel("Test Error (in %)",fontsize = fontsize+10)
    axes.set_title(title, fontsize = fontsize+10)


#-------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
which = "shear_layer"

plot = True
if plot:
    device = "cpu"
    batch = 1
else:
    batch = 32

#--------------------------

FNO = []
CNO = []

if  which == "shear_layer":
    
    folder_FNO  = "CNO2FNO2SELECTED/FNO/fno_best_navier_stokes"
    folder_CNO  = "CNO2FNO2SELECTED/CNO/cno_best_navier_stokes"
    
    N = 128

if not plot:
    
    modelFNO = torch.load(folder_FNO + "/model.pkl", map_location=torch.device(device))
    modelCNO = torch.load(folder_CNO + "/model.pkl", map_location=torch.device(device))
    modelFNO.device = device
    modelCNO.device = device

    for s_test in range(32, 129, 8):
        
        data_loader_CNO = load_data(folder_CNO, "CNO", device, which, 64, batch, 1, s_test)
        
        E_CNO = error_distribution(which_model = "CNO", model = modelCNO, testing_loader = data_loader_CNO, p = 1, N = N, device = device, s = s_test, s_train = 64, which = which)
        print("done CNO")
        CNO.append(np.median(E_CNO))

        print(s_test, np.median(E_CNO))

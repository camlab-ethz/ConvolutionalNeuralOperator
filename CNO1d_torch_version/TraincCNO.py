import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
from CNO1d import CNO1d
import matplotlib.pyplot as plt


# In this script, we approxiamte solution of the 1d Allen-Cahn equation

def main():
    n_train = 100 # number of training samples

    # Load the data
    # - data/AC_data_input.npy
    # - data/AC_data_output.npy

    # We will decrease the resolution to s = 256, for more convenient training

    x_data = torch.from_numpy(np.load("data/AC_data_input.npy")).type(torch.float32)
    y_data = torch.from_numpy(np.load("data/AC_data_output.npy")).type(torch.float32)

    x_data = x_data.permute(0,2,1)
    y_data = y_data.unsqueeze(1)
    s = 256
    x_data = F.interpolate(x_data.unsqueeze(2), size = (1, s), mode = "bicubic")[:,:,0]
    y_data = F.interpolate(y_data.unsqueeze(2), size = (1, s), mode = "bicubic")[:,:,0]


    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:, :]
    output_function_test = y_data[n_train:, :]

    batch_size = 10

    training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size, shuffle=False)

    
    #---------------------
    # Define the hyperparameters and the model:
    #---------------------
    
    learning_rate = 0.001
    epochs = 50
    step_size = 15
    gamma = 0.5

    N_layers = 4
    N_res    = 4
    N_res_neck = 4
    channel_multiplier = 16

    cno = CNO1d(in_dim = 2,                                    # Number of input channels.
                out_dim = 1,                                   # Number of input channels.
                size = s,                                      # Input and Output spatial size (required )
                N_layers = N_layers,                           # Number of (D) or (U) blocks in the network
                N_res = N_res,                                 # Number of (R) blocks per level (except the neck)
                N_res_neck = N_res_neck,                       # Number of (R) blocks in the neck
                channel_multiplier = channel_multiplier,       # How the number of channels evolve?
                use_bn = False)

    #-----------
    # TRAIN:
    #-----------
    
    optimizer = AdamW(cno.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    l = nn.L1Loss()
    freq_print = 1
    for epoch in range(epochs):
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set):
            optimizer.zero_grad()
            output_pred_batch = cno(input_batch)
            loss_f = l(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        with torch.no_grad():
            cno.eval()
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(testing_set):
                output_pred_batch = cno(input_batch)
                loss_f = (torch.mean((abs(output_pred_batch - output_batch))) / torch.mean(abs(output_batch))) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, " ######### Relative L1 Test Norm:", test_relative_l2)


    
if __name__ == "__main__":
    main()

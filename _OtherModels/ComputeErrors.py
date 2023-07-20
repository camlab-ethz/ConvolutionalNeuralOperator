import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

model = sys.argv[1]
which = sys.argv[2]
import ast

if model == "DON":
    from Problems.BenchmarksDON import EquationModel, ShearLayer

elif model == "UNet":
    from Problems.BenchMarksUNet import EquationModel, ShearLayer

elif model == "ResNet":
    from Problems.BenchmarksResNet import EquationModel, ShearLayer

elif model == "ResNet2":
    from Problems.BenchMarkResNet2 import EquationModel, ShearLayer

print(which, model)
model_architecture = dict()
folder = "SELECTED_MODELS/Best_" + which + "_" + model
with open(folder + "/net_architecture.txt") as f:
    for line in f:
        key_values = line.replace("\n", "").split(",")
        key = key_values[0]
        value = ast.literal_eval(key_values[1])
        model_architecture[key] = int(value)

sample_path = folder

plot=False
if os.path.isfile(folder + "/model.pkl"):
    mod = torch.load(folder + "/model.pkl", map_location=torch.device('cpu')).cpu()
for insample in [True, False]:
    print("Insample: ", insample)
    if which == "shear_layer":
        example = ShearLayer(model_architecture, "cpu", 1, 750, insample=insample)
    elif which == "poisson":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=1024, which_data="poisson", insample=insample)
    elif which == "wave":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=512, which_data="wave", insample=insample)
    elif which == "allen_cahn":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=256, which_data="allen_cahn", insample=insample)
    elif which == "cont_t":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=512, which_data="cont_t", insample=insample)
    elif which == "discont_t":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=512, which_data="discont_t", insample=insample)
    elif which == "airfoil":
        example = EquationModel(model_architecture, "cpu", 1, training_samples=512, which_data="airfoil", insample=insample)

    testing_loader = example.test_loader_2

    error_vec = np.zeros((len(testing_loader), 2))
    for p in [1, 2]:
        print("p ", p)
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(testing_loader):

                if model != "DON":

                    # (inputs.shape)
                    prediction = mod(inputs)


                else:

                    prediction = mod(inputs, example.trunk_inputs)

                if which == "airfoil":
                    outputs[inputs == 1] = 1
                    prediction[inputs == 1] = 1
                err = (torch.mean(abs(prediction[0, 0, :, :] - outputs[0, 0, :, :]) ** p) / torch.mean(abs(outputs[0, 0, :, :]) ** p)) ** (1 / p) * 100
                error_vec[i, p - 1] = err.item()


                if plot:
                    if i<3:
                        fig, ax = plt.subplots(1,1)
                        '''ax[0].imshow(outputs[0, 0], origin="lower", cmap="gist_ncar", extent=[0, 1, 0, 1], interpolation="bicubic", vmin=0, vmax=1)
                        ax[0].set_xlabel(r'$x$', fontsize=12)
                        ax[0].set_ylabel(r'$y$', fontsize=12)'''
                        im1=ax.imshow(outputs[0, 0], origin="lower", cmap="gist_ncar", extent=[0, 1, 0, 1], interpolation="bicubic", vmin=0, vmax=1)
                        ax.set_xlabel(r'$x$', fontsize=10)
                        ax.set_ylabel(r'$y$', fontsize=10)
                        plt.colorbar(im1)
                        # Adjust the margins
                        plt.tight_layout()
                        plt.savefig("True_"+model+"_ "+ which + "_"+str(i)+str(insample)+".png", dpi=200, bbox_inches='tight')
                    else:
                        break

        # %%

    fname = "sum_errors_complete_in" if insample else "sum_errors_complete_out"
    with open(folder + '/' + fname + '.txt', 'w') as file:
        file.write("Median L1:" + str(np.median(error_vec[:, 0])) + "\n")
        file.write("25 Quantile L1:" + str(np.quantile(error_vec[:, 0], 0.25)) + "\n")
        file.write("75 Quantile L1:" + str(np.quantile(error_vec[:, 0], 0.75)) + "\n")
        file.write("Std L1:" + str(np.std(error_vec[:, 0])) + "\n")
        file.write("Mean L1:" + str(np.mean(error_vec[:, 0])) + "\n")

        file.write("Median L2:" + str(np.median(error_vec[:, 1])) + "\n")
        file.write("25 Quantile L2:" + str(np.quantile(error_vec[:, 1], 0.25)) + "\n")
        file.write("75 Quantile L2:" + str(np.quantile(error_vec[:, 1], 0.75)) + "\n")
        file.write("Std L2:" + str(np.std(error_vec[:, 1])) + "\n")
        file.write("Mean L2:" + str(np.mean(error_vec[:, 1])) + "\n")

import itertools
import os
import sys

import numpy as np

np.random.seed(0)

cluster = "false"
sbatch = False

all_training_properties = {
    "learning_rate": [0.0005, 0.001],
    "weight_decay": [0, 1e-6],
    "scheduler_step": [10],
    "scheduler_gamma": [1, 0.98],
    "epochs": [1000],
    "batch_size": [10],
    "exp": [1],
    "training_samples": [890],  # useless
}

all_model_architecture = {
    "FourierF": [0],
    "retrain": [4, 76, 134],
    "channels": [8, 16, 32, 64]
}

which_example_list = ["poisson", "wave", "allen_cahn", "cont_t", "discont_t", "shear_layer", "airfoil"]

for which_example in which_example_list:

    folder_name = "ModelSelectionUNet_" + which_example
    ndic = {**all_training_properties,
            **all_model_architecture}

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    settings = list(itertools.product(*ndic.values()))
    idx = np.random.choice(len(settings), 30, replace=False)
    setting_new = list()
    for i in range(idx.shape[0]):
        setting_new.append(settings[idx[i]])

    i = 0
    for setup in setting_new:
        # time.sleep(10)
        # print(setup)
        training_properties_ = {
        }
        j = 0
        for k, key in enumerate(all_training_properties.keys()):
            training_properties_[key] = setup[j]
            j = j + 1

        model_architecture_ = {
        }
        for k, key in enumerate(all_model_architecture.keys()):
            model_architecture_[key] = setup[j]
            j = j + 1

        arguments = list()
        arguments.append(folder_name)
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            if sbatch:
                arguments.append("\\\"" + str(training_properties_) + "\\\"")
            else:
                arguments.append("\'" + str(training_properties_).replace("\'", "\"") + "\'")

        else:
            arguments.append(str(training_properties_).replace("\'", "\""))

        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            if sbatch:
                arguments.append("\\\"" + str(model_architecture_) + "\\\"")
            else:
                arguments.append("\'" + str(model_architecture_).replace("\'", "\"") + "\'")

        else:
            arguments.append(str(model_architecture_).replace("\'", "\""))

        arguments.append(which_example)

        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            if cluster == "true":
                string_to_exec = "sbatch --time=16:00:00 -n 1 -G 1 --mem-per-cpu=8192 --wrap=\"python3 TrainUNet.py"
            else:
                string_to_exec = "python3 TrainUNet.py "
            for arg in arguments:
                string_to_exec = string_to_exec + " " + arg
            if cluster == "true":
                string_to_exec = string_to_exec + " \""
            print(string_to_exec)
            os.system(string_to_exec)
        i = i + 1

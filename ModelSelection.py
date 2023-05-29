import itertools
import os
import sys

import numpy as np

np.random.seed(0)

folder_name = "MODEL_SELECTION_Wave_TMP"
cluster = "false"


all_training_properties = {
    "learning_rate": [0.001, 0.0005],
    "weight_decay": [1e-10],
    "scheduler_step": [10],
    "scheduler_gamma": [0.99, 0.95],
    "epochs": [300],
    "batch_size": [32, 16],
    "exp": [1, 2],
    "training_samples": [1024],
}


all_model_architecture = {
    "in_size": [64],
    "cutoff_den": [2.00001],
    "N_layers": [6,8,10],
    "N_res": [2, 4, 6, 8, 10],
    "filter_size": [6],
    "radial_filter": [0],
    "kernel_size": [3],
    "FourierF": [0],
    "channel_multiplier": [16, 32, 48],
    "half_width_mult": [1],
    "lrelu_upsampling": [2],
    "retrain": [4]
}

all_training_properties = {
    "learning_rate": [0.001],
    "weight_decay": [1e-10],
    "scheduler_step": [10],
    "scheduler_gamma": [0.99],
    "epochs": [300],
    "batch_size": [32, 16],
    "exp": [1, 2],
    "training_samples": [1024],
}


all_model_architecture = {
    "in_size": [64],
    "cutoff_den": [2.00001],
    "N_layers": [4, 6],
    "N_res": [8, 10],
    "filter_size": [6],
    "radial_filter": [0],
    "kernel_size": [3, 5],
    "FourierF": [0],
    "res_len": [2, 3],
    "channel_multiplier": [32],
    "half_width_mult": [1],
    "lrelu_upsampling": [2],
    "retrain": [4]
}



which_example = "wave_0_5"

ndic = {**all_training_properties,
        **all_model_architecture}

print(folder_name)
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*ndic.values()))

sbatch = False

i = 0
for setup in settings:
    # time.sleep(10)
    print(setup)

    folder_path = "\'" + folder_name +"/"+str(setup[7])+"Setup_" + str(i) + "\'"

    #folder_path = "$SCRATCH"+"\'" + folder_name + "/Setup_" + str(i) + "\'"

    print(folder_path)
    print("###################################")
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
    arguments.append(folder_path)
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
            if sbatch:
                string_to_exec = "sbatch --time=3:00:00 -n 16 --mem-per-cpu=512 --wrap=\"python3 TrainCNO.py"
            else:
                string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 TrainCNO.py"
        else:
            string_to_exec = "python3 TrainCNO.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        if  cluster =="true" and sbatch:
            string_to_exec = string_to_exec + " \""
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1

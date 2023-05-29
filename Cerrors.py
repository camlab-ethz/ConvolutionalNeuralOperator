import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

np.random.seed(42)

models = ["DON", "UNet", "ResNet"]
models = ["UNet"]
which_example_list = ["poisson", "wave", "allen_cahn", "cont_t", "discont_t", "shear_layer", "airfoil"]

#os.mkdir("FinalModel")

for model in models:
    for which in which_example_list:

        str_to_exec = r'sbatch --mem-per-cpu=32000 --wrap="python3 ComputeErrors.py ' +model + r' ' + which + r'"'
        print(str_to_exec)
        os.system(str_to_exec)

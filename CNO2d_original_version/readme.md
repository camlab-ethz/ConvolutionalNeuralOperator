# CNO2d - original version
#### In this folder, one can find the original CNO code from the NeurIPS 2023.

## Notes and Instructions
1. **Training a CNO is slow on CPU. We suggest the training to be run on a GPU!**
2. To run the original CNO code, one needs the CUDA toolkit 11.1 or later. This toolkit is **NOT** the same as cudatoolkit from Conda. Please visit the page https://developer.nvidia.com/cuda-toolkit for installation!
3. To run the CNO code on Linux, one needs GCC 7 or later compiler.
	To run the CNO code on Windows, one needs Visual Studio compiler.
4. To run the CNO code, one needs [ninja](https://pypi.org/project/ninja/) build system. Conda installation can be found [here](https://anaconda.org/conda-forge/ninja).

Implementation of the filters from the original CNO code is borrowed from the paper *Alias-Free Generative Adversarial Networks (StyleGAN3)* -- see their [github page](https://github.com/NVlabs/stylegan3).

## Training
The CNO models can be trained by running the python scripts

	 TrainCNO.py

Important hyperparameters related to the CNO architecture are:

| Parameter | Meaning |
| ------ | ------ |
| N_layers | number of up/downsampling blocks |
| channel_multiplier | regulates the number of channels in the network |
| N_res | number of residual blocks in the middle networks |
| N_res_neck |  number of residual blocks in the bottleneck |
| in_size |  resolution of the computational grid |

Other parameters may be kept unchanged,

To select the benchmark experiment for CNO to be trained, the variable *which_example"*in a corresponding script TrainCNO.py should have one of the following values:

| which_example | PDE |
| ------ | ------ |
| poisson | Poisson equation |
| wave_0_5 | Wave equation |
| cont_tran | Smooth Transport |
| disc_tran |  Discontinuous Transport |
| allen | Allen-Cahn equation |
| shear_layer | Navier-Stokes equations |
| airfoil | Compressible Euler equations |
| darcy | Darcy Flow |

The file 

	Problems/CNOBenchmark.py

corresponds to the dataloader for CNO model

## Hyperparameters Grid/Random Search
Cross validation for each model can be run with:

	python3 ModelSelectionCNO.py

The hyperparameters of the best-performing models reported in the Supplementary Materials are obtained in this way.

#### Note
If a slurm-base cluster is available, set sbatch=True and cluster="true" in the scripts. We ran the codes on a local cluster (Euler cluster).

## Error Computations

To compute the relative L1 median errors of the CNO and FNO models, one scould run the scripts "ErrorDistribution.py".

In the "ErrorDistribution.py" file, one should select the variable "which", corresponding to a benchmark experiment.

In the same file, one can set a variable "plot = True" to plot a random sample and predictions for the **CNO and FNO models**.

One can also set "plot = False" to compute the errors for the CNO and FNO models. By selecting "in_dist = False", one obtains out-of-distribution test errors. 


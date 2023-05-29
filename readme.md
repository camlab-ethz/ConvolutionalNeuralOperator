# Convolutional Neural Operators for robust and accurate learning of PDEs

This repository is the official implementation of the paper **Convolutional Neural Operators for robust and accurate learning of PDEs**

![alt text](/figures/fig.png)


## Requirements
The code is based on python 3 (version 3.7) and the packages required can be installed with

	python3 -m pip install -r requirements.txt


#### Note:

1. **Training a CNO is slow on CPU. We suggest the training to be run on a GPU!**

2. To run the CNO code, one needs the CUDA toolkit 11.1 or later. This toolkit is **NOT** the same as cudatoolkit from Conda. Please visit the page https://developer.nvidia.com/cuda-toolkit for installation!
	
3.	To run the CNO code on Linux, one needs GCC 7 or later compiler.
	To run the CNO code on Windows, one needs Visual Studio compiler.

4.	To run the CNO code, one needs ninja build system.

Implementation of the filters is borrowed from the paper *Alias-Free Generative Adversarial Networks (StyleGAN3)*. 
Their official github page is https://github.com/NVlabs/stylegan3.

## Source Data
We cover instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations. Data can be downloaded from https://zenodo.org/record/7963379 (~10GB).

Alternatively, run the script `download_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).


	python3 download_data.py


The "data.zip" needs to be unzipped.

## Models Training
Each of the baselines described in the paper can be trained by running the python scripts 


	Train**.py


where ** holds for:

	- CNO:    CNO model
	- FNO:    FNO model
	- DON:    DeepONet model
	- UNet:   UNet model
    - ResNet: Feedforward neural network (FFNN) with residual connections

The models' hyperparameter can be specified in the corresponding python scripts as well.

To select the benchmark experiment for FNO and CNO to be trained, the variable "which_example" in a corresponding script Tran**.py should have one of the following values:

    poisson             : Poisson equation 
    wave_0_5            : Wave equation
    cont_tran           : Smooth Transport
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation
    shear_layer         : Navier-Stokes equations
    airfoil             : Compressible Euler equations


To select the benchmark experiment for UNet, DeepONet and FFNN to be trained, the variable "which_example" in a corresponding script Tran**.py should have one of the following values:

    poisson             : Poisson equation 
    wave                : Wave equation
    cont_t              : Smooth Transport
    disc_t              : Discontinuous Transport
    allen_cahn          : Allen-Cahn equation
    shear_layer         : Navier-Stokes equations
    airfoil             : Compressible Euler equations


#### Note



The following files correspond to:

	Problems/Benchmark.py :       Dataloader for CNO model
	Problems/FNOBenchmark.py :    Dataloader for FNO model
	Problems/BenchMarksUNet.py:   Dataloader for UNet model
	Problems/BenchmarksDON.py:    Dataloader for DeepONet model
	Problems/BenchmarksResNet.py: Dataloader for FFNN model
	

## Hyperparameters Grid/Random Search
Cross validation for each model can be run with:


	python3 ModelSelection**.py


where ** correspond to a model, as noted above.

The hyperparameters of the best-performing models reported in the Supplementary Materials are obtained in this way.


#### Note
If a slurm-base cluster is available, set sbatch=True and cluster="true" in the scripts. We ran the codes on a local cluster.

## Pretrained Models
The models trained and used to compute the errors in Table 1 can be downloaded by running:


	python3 download_models.py


The compressed folder has to be unzipped!

Models can also be downloaded from https://zenodo.org/record/7963379 .

## Error Computations

The errors of the best performing CNO, FNO and UNet models (Table 1) can be computed by running the scripts "ErrorDistribution.py".

In the "ErrorDistribution.py" file, one should select the variable "which", corresponding to a benchmark experiment. It should have one of the following values:

    poisson             : Poisson equation 
    wave_0_5            : Wave equation
    cont_tran           : Smooth Transport
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation
    shear_layer         : Navier-Stokes equations
    airfoil             : Compressible Euler equations

In the same file, one can set a variable "plot = True" to plot a random sample and predictions for the CNO, FNO and UNet models.
One can also set "plot = False" to compute the errors for the CNO, FNO and UNet models. By selecting "in_dist = False", one obtains out-of-distribution test errors. 

The errors of the best performing FFNN, DeepONet or Unet (Table 1) can be computed by running the scripts "ComputeErrors.py".

	python3 ComputeErrors.py model which_example

with model being either "ResNet", "DON", "UNet" and which being:

    poisson             : Poisson equation 
    wave                : Wave equation
    cont_t              : Smooth Transport
    disc_t              : Discontinuous Transport
    allen_cahn          : Allen-Cahn equation
    shear_layer         : Navier-Stokes equations
    airfoil             : Compressible Euler equations


### Varying resolution

The errors of the best performing CNO, FNO and UNet models for different resolutions can be computed by running the scripts "ErrorDistribution_VaryingResolution.py".

In the "Error_Distribution_VaryingResolution.py" file, one should select the variable "which", corresponding to a benchmark experiment. It should have one of the following values:

    poisson             : Poisson equation 
    wave                : Wave equation
    shear_layer         : Navier-Stokes equations

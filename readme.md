# Convolutional Neural Operators for robust and accurate learning of PDEs

This repository is the official implementation of the paper **Convolutional Neural Operators for robust and accurate learning of PDEs** (see https://arxiv.org/pdf/2302.01178.pdf). The paper was presented at **NeurIPS 2023**.

**Representative PDE Benchmarks (RPB) are available at** https://zenodo.org/records/10406879 !

**Read our blog at** https://link.medium.com/Mht8Th5OhFb !

![alt text](/figures/diag.png)

The CNO is tested on a novel set of benchmarks, termed as Representative PDE Benchmarks (RPB), that span across a variety of PDEs ranging from linear elliptic and hyperbolic to nonlinear parabolic and hyperbolic PDEs, with possibly multiscale solutions. The CNO is either on-par or outperformed the tested baselines on all the benchmarks, both when testing in-distribution as well as in out-of-distribution testing.

<p align="center">
 <img src="/figures/table.png" width="750"/>
</p>
<p align="center">
    <em>Relative median L¹ test errors, for both in- and out-of-distribution testing, for different benchmarks and models..</em>
</p>
<br />

We assess the test errors of the CNO and other baselines at different testing resolutions notably, for the Navier-Stokes equations benchmarks. We observe that in this case, the CNO is the only model that demonstrates approximate error invariance with respect to test resolution.

<p align="center">
 <img src="/figures/resolution_NS.png" width="500"/>
</p>
<p align="center">
    <em>The CNO model has almost constant testing error across different resolutions (Navier-Stokes).</em>
</p>
<br />


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

<br />

**Note: To train or evaluate models other than CNO, please move the required files/scripts/modules from the folder _OtherModels to the main folder.**


## Source Data
We cover instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations and Darcy flow. Data can be downloaded from https://zenodo.org/records/10406879 (~2.4GB).

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
	...

The models' hyperparameter can be specified in the corresponding python scripts as well.

To select the benchmark experiment for FNO and CNO to be trained, the variable "which_example" in a corresponding script Tran**.py should have one of the following values:

    poisson             : Poisson equation 
    wave_0_5            : Wave equation
    cont_tran           : Smooth Transport
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation
    shear_layer         : Navier-Stokes equations
    airfoil             : Compressible Euler equations
    darcy               : Darcy Flow

#### Note



The following files correspond to:

	Problems/CNOBenchmark.py :            Dataloader for CNO model
	_OtherBenchamrks/FNOBenchmark.py :    Dataloader for FNO model
	_OtherBenchamrks/BenchmarksDON.py:    Dataloader for DeepONet model
        ...	

## Hyperparameters Grid/Random Search
Cross validation for each model can be run with:


	python3 ModelSelection**.py


where ** correspond to a model, as noted above.

The hyperparameters of the best-performing models reported in the Supplementary Materials are obtained in this way.


#### Note
If a slurm-base cluster is available, set sbatch=True and cluster="true" in the scripts. We ran the codes on a local cluster (Euler cluster).


## Error Computations

To compute the relative L1 median errors of the CNO and FNO models, one scould run the scripts "ErrorDistribution.py".

In the "ErrorDistribution.py" file, one should select the variable "which", corresponding to a benchmark experiment. 

In the same file, one can set a variable "plot = True" to plot a random sample and predictions for the CNO and FNO models.
One can also set "plot = False" to compute the errors for the CNO and FNO models. By selecting "in_dist = False", one obtains out-of-distribution test errors. 


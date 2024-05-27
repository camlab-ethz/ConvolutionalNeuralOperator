# Convolutional Neural Operators for robust and accurate learning of PDEs

- This repository is the official implementation of the paper **Convolutional Neural Operators for robust and accurate learning of PDEs** (see https://arxiv.org/pdf/2302.01178.pdf)
- The paper was presented at **NeurIPS 2023**
- **Representative PDE Benchmarks (RPB) are available at** https://zenodo.org/records/10406879
- **Read our blog at** https://link.medium.com/Mht8Th5OhFb

The CNO is tested on a novel set of benchmarks, termed as Representative PDE Benchmarks (RPB). The CNO is either on-par or outperformed the tested baselines on all the benchmarks, both when testing in-distribution as well as in out-of-distribution testing.

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

## Code Instructions:

1. **The original CNO code from NeurIPS 2023 is located in the folder [CNO2d_original_version](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_original_version)**
    - All the instructions for this version can be found in the _readme.md_ file in the folder
    - The code is more complex to configure compared to the vanilla CNO code (see below)

2. **Vanilla CNO2d and CNO1d versions are located in the folders [CNO2d_vanilla_torch_version](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_vanilla_torch_version) and [CNO1d_vanilla_torch_version](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO1d_vanilla_torch_version)**
    - All the instructions for these versions can be found in the _readme.md_ files in the folders
    - The models are termed as "vanilla CNO" as the interpolation filters cannot be manually designed
    - The codes do not utilize the CUDA kernel, making them simple to configure

3. **Codes for Time-Dependent CNO2d** - [CNO2d_time_dependent_pde](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_time_dependent_pde) - **COMING SOON**

4. **Codes for the other baselines are located in the folder [_OtherModels](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/_OtherModels)**


## Datasets
We cover instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations and Darcy flow. Data can be downloaded from https://zenodo.org/records/10406879 (~2.4GB).

Alternatively, run the script `download_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).


	python3 download_data.py


The "data.zip" needs to be unzipped.


# Convolutional Neural Operators for robust and accurate learning of PDEs

- This repository is the official implementation of the paper [**Convolutional Neural Operators for robust and accurate learning of PDEs**](https://arxiv.org/pdf/2302.01178.pdf)
- The paper was presented at **NeurIPS 2023**
- **Representative PDE Benchmarks (RPB) are available at [this link](https://zenodo.org/records/10406879)** 
- **Read our blog about CNOs at [this link](https://link.medium.com/Mht8Th5OhFb)** 
- This repository also covers CNO codes used in the paper [**Poseidon: Efficient Foundation Models for PDEs**](https://arxiv.org/abs/2405.19101)
- Vist [**this link**](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_time_dependent_%26_foundation_model) for **time-dependent CNO** and **CNO - foundation model**
- Visit [**Poseidon github page**](https://github.com/camlab-ethz/poseidon)


The CNO is tested on a novel set of benchmarks, termed as Representative PDE Benchmarks (RPB). The CNO is either on-par or outperformed the tested baselines on all the benchmarks, both when testing in-distribution as well as in out-of-distribution testing.

<p align="center">
 <img src="/figures/table.png" width="750"/>
</p>
<p align="center">
    <em>Relative median L¹ test errors, for both in- and out-of-distribution testing, for different benchmarks and models.</em>
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

3. **Codes for Time-Dependent CNO2d are located in the folder** - [CNO2d_time_dependent_&_foundation_model](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_time_dependent_%26_foundation_model)
   - The Time-Dependent CNO and CNO - Foundation Model codes are used in the paper [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101)
   - All the instructions for the Time-Dependent CNO can be found in the _readme.md_ file in the folder

4. **Codes for CNO - Foundation Model are located in the folder** - [CNO2d_time_dependent_&_foundation_model](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_time_dependent_%26_foundation_model)
   - All the instructions for the CNO - Foundation Model can be found in the _readme.md_ file in the folder
   - Codes for finetuning the CNO-FM are also located in the folder.
   - One can download the weights CNO-Foundation Model (109M) on [this link](https://zenodo.org/records/11401801).

5. **Codes for the other baselines are located in the folder [_OtherModels](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/_OtherModels)**


## Datasets
We cover instances of the Poisson, Wave, Navier-Stokes, Allen-Cahn, Transport and Compressible Euler equations and Darcy flow. Data can be downloaded from https://zenodo.org/records/10406879 (~2.4GB).

Alternatively, run the script `download_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).

	python3 download_data.py

## Poseidon: Efficient Foundation Models for PDEs
We also provide all datasets used in the paper [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101) on the 🤗 Hub. You can download them from the respective collections: 
- [🤗 Hub – Pretraining Datasets](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- [🤗 Hub – Downstream Tasks](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)

### Please also visit [Poseidon github page](https://github.com/camlab-ethz/poseidon).

## Citation

If you use our models, code, or datasets, please consider citing our paper:

```bibtex
@misc{CNO,
      title={Convolutional Neural Operators for robust and accurate learning of PDEs}, 
      author={Bogdan Raonić and Roberto Molinaro and Tim De Ryck and Tobias Rohner and Francesca Bartolucci and Rima Alaifari and Siddhartha Mishra and Emmanuel de Bézenac},
      year={2023},
      eprint={2302.01178},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

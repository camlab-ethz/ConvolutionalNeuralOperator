
# Time-Dependent CNO2d & CNO-Foundation Model
- ### In this folder, one can find the implementation of Time-Dependent Convolutional Neural Operator, introduced in the paper [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101).  See the official [github page](https://github.com/camlab-ethz/poseidon/blob/main/README.md) of the paper. 
- ### Moreover, one can train, evaluate and **finetune** the CNO - Foundation Model (109M), introduced in the same paper. The weights of CNO-FM are available [here](https://zenodo.org/records/11401801).

## Time-Dependent CNO
Time variable is continuosly embedded into CNO by  **lead-time conditioned instance normalization**. By continuously integrating time into a Neural Operator, the model can be evaluated at any point in time, including beyond the training times. We also observed that *including time* as an additional, constant input channel of the CNO slightly enhances its performance.

## Notes and Instructions
1. **Training a CNO is slow on CPU. We suggest the training to be run on a GPU!**
2. To run the Time-Dependent CNO code, one needs the CUDA toolkit 11.1 or later. This toolkit is **NOT** the same as cudatoolkit from Conda. Please visit the page https://developer.nvidia.com/cuda-toolkit for installation!
3. To run the Time-Dependent  CNO code on Linux, one needs GCC 7 or later compiler.
	To run the Time-Dependent CNO code on Windows, one needs Visual Studio compiler.
4. To run the Time-Dependent CNO code, one needs [ninja](https://pypi.org/project/ninja/) build system. Conda installation can be found [here](https://anaconda.org/conda-forge/ninja).
5. One also needs the latest verison of [pytorch lightning](https://lightning.ai/docs/pytorch/stable/).
6. Implementation of the filters is borrowed from the paper *Alias-Free Generative Adversarial Networks [(StyleGAN3)](https://github.com/NVlabs/stylegan3)*.

#### Note: If you are unable to install cudatoolkit or other libraries, you can select *cno_lrelu_torch* or *lrelu* as the activation function.

## Training
The CNO models can be trained by running the python scripts

	 TrainCNO_time_L.py

Important hyperparameters related to the CNO architecture are:

| Parameter | Meaning |
| ------ | ------ |
| N_layers | number of up/downsampling blocks |
| channel_multiplier | regulates the number of channels in the network |
| N_res | number of residual blocks in the middle networks |
| N_res_neck |  number of residual blocks in the bottleneck |
| in_size |  resolution of the computational grid |
| batch_norm | use simple BN? 1 = use it, 0 = do not use it |
| is_time | use conditional IN/LN/BN for time? 1 = use it & **do not** use simple BN |
| nl_dim | if is_time = 1 - '23'is CIN, '023' is CBN, '123'is CLN |
| activation | cno_lrelu, cno_lrelu_torch or lrelu |
|is_att| use attention in the bottleneck? default is 0 |

Important parameters related to the CNO training are:
| Parameter | Meaning |
| ------ | ------ |
| time_steps | number of time steps in the training trajectory |
| dt | What is the time step?  e.g. 2 means take every other step |
| time_input | Should we include time in the input channels (1 = include)|
| allowed | What kind of training? all2all, one2all or one (AR train)|

### Benchmarks

We provide all datasets used in the paper on the 🤗 Hub. You can download them from the respective collections: 
- [🤗 Hub – Pretraining Datasets](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- [🤗 Hub – Downstream Tasks](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)

 Before running any script, please add the relevant paths of your datasets in **DataLoaders/load_utils.py** and **DataLoaders/CNO_TimeLoaders.py** scripts.

To select the benchmark experiment for CNO to be trained, the variable **which_example** in a corresponding script *TrainCNO_time_L.py* should  be selected. It needs to have one of the following values:

| which_example | 🤗 Hub/Paper Identifier |
| ------ | ------ |
| ns_brownian | NS-BB | 
| ns_pwc | NS-PwC | 
| ns_gauss | NS-Gauss | 
| ns_sin | NS-Sines |
| ns_vortex | NS-SVS | 
| ns_shear | NS-SL | 
| ns_pwc_t | NS-Tracer-PwC | 
| eul_kh | CE-KH | 
| eul_riemann | CE-RP | 
| eul_riemann_kh | CE-RPUI | 
| eul_riemann_cur | CE-CRP| 
| eul_gauss | CE-Gauss | 
| rich_mesh | CE-RM | 
| rayl_tayl | GCE-RT | 
| kolmogorov | FNS-KF | 
| wave_seismic | Wave-Layer | 
| wave_gauss | Wave-Gauss | 
| allen_cahn | ACE | 
| airfoil | SE-AF | 
| poisson_gauss | Poisson-Gauss | 
| helmholtz | Helmholtz | 

## Finetunning CNO-FM

One can download the weights CNO-Foundation Model (109M) on [this link](https://zenodo.org/records/11401801).  Model specifications are given in the paper [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101).

 One can **fine-tune** the CNO-FM with the script 
 
	 CNO_FineTune.py

One must select an experiment from the list above and modify the relevant parameters in the script. Important parameters related to the finetuning CNO are:

| Parameter | Meaning |
| ------ | ------ |
| is_different_dim | Are the dimensions different from the pretraining (i.e. out-of-context)? 1 = yes |
| in_dim_tune |  If yes, what is the input dimension?|
| out_dim_tune | If yes, what is the output dimension?|
| steps | How many time steps should we include in the dataset? |

**NOTE**: Input dimension of the CNO-FM is 5 (density, x-velocity, y-velocity, pressure, time). Output dimension of the CNO-FM is 4  (density, x-velocity, y-velocity, pressure)

## Running your own experiments

The file 

	DataLoaders/CNO_TimeLoaders.py

corresponds to the dataloaders for the CNO model.

- To run your own experiment, you should write your own class in the *CNO_TimeLoaders.py*.
- Once the class is coded, you should add your loader in the *DataLoaders/load_utils.py* script and *DataLoaders/all_experiments.json* file
- Once this is done, you should be able load your data in the *TrainCNO_time_L.py* file.



## Error Computations

To compute the relative L1 median errors of the CNOmodels, one scould run the scripts *TestCNO_ALL.py*.  One should set relevant parameters to theit true values to be able to run the script.


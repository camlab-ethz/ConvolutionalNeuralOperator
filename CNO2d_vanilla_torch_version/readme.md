
# CNO2d - Vanilla Version

- The CNO2d code has been modified from a tutorial featured in the ETH Zurich course "AI in the Sciences and Engineering."
- Git page for this course: https://github.com/bogdanraonic3/AI_Science_Engineering 

## What is different?
- For up/downsampling, the antialias interpolation functions from the  torch library are utilized, limiting the ability to design your own low-pass filters at present.

- The performance of CNO2d remains commendable.

## Training

A training script is available, offering a solid foundation for personal projects. 

The CNO models can be trained by running the python scripts

	 TrainCNO.py

Important hyperparameters related to the CNO architecture are:

| Parameter | Meaning |
| ------ | ------ |
| N_layers | number of up/downsampling blocks |
| channel_multiplier | regulates the number of channels in the network |
| N_res | number of residual blocks in the middle networks |
| N_res_neck |  number of residual blocks in the bottleneck |
| s |  resolution of the computational grid |

## Data Loaders

- One can use loaders similar to the ones from the [original CNO version](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_original_version).

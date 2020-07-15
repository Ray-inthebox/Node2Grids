# N2G in Pytorch
------
We provide Pytorch implementations for the paper "TBD". You need to install Pytorch and other related python packages. the dependencies are listed below.
```
- scipy
- numba
- numpy
- sklearn
```
# Data Preparation
We provide the datasets including cora, citeseer, pubmed. For the Amazon2M, we will upload it to github later, or you can download it from [here](https://drive.google.com/drive/folders/1qWL76l7wYcESLfYNg0KbNjk22DEP_CTQ?usp=sharing "here").
# Run the Code
For all dataset, try
```
python main.py
```
If you have CUDA devices, try
```
CUDA_VISIBLE_DEVICES='YOUR CUDA DEVICES' python main.py
```

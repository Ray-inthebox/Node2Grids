# Node2Grids in Pytorch
------
We provide Pytorch implementations for the paper "[An Uncoupled Training Architecture for Large Graph Learning](https://arxiv.org/abs/2003.09638 "TBD")". You need to install Pytorch and other related python packages. the dependencies are listed below.
```
- scipy
- numba
- numpy
- sklearn
```
# Data Preparation
We provide the datasets including cora, citeseer, pubmed. We use the ogbn-products dataset for large-scale dataset testing, you can download it [here](https://ogb.stanford.edu/docs/nodeprop/ "here").
# Run the Code
For all dataset, try
```
python main.py
```
If you have CUDA devices, try
```
CUDA_VISIBLE_DEVICES='YOUR CUDA DEVICES' python main.py
```
If you try to use large scale dataset, a parallel version of preprocessing is provided. Specify the param *num_woker* to make parallelable.
```
findTopK(adj, idx, k, nodesdegree, nums_worker=2 (or larger, which is default to 1))
```

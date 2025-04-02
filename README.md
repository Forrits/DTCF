# DTCF

## Installation
* CentOS Linux release 7.3.1611
* Python 3.6.13
* CUDA 9.2
* PyTorch 1.9.0
* medpy 0.4.0
* tqdm，h5py

## Getting Started
Please change the database path and data partition file in the corresponding code.
### Dataset
[LA data](https://drive.google.com/drive/folders/1_LObmdkxeERWZrAzXDOhOJ0ikNEm0l_l?usp=sharing)  
[Pancreas data](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz?usp=sharing)
### Training
`pyhon train_DTCF.py`
### Evaluation
`pyhon test_DTCF.py`

## Acknowledgement
We build the project based on UA-MT,SASSNet,DTC,MCF
Thanks for their contribution.

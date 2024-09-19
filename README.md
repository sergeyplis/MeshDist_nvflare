# MeshDist_nvflare
MeshNet distributed learning by using the NVFlare framework.  

This work is in progress.. 



## Quick Setup Guide

### 1. Clone the Repository

First, clone the repository from GitHub or the relevant location to your machine:

git clone https://github.com/Mmasoud1/MeshDist_nvflare.git

cd MeshDist_NVFlare



### 2. Check for PyTorch with CUDA Support 

In the terminal, check if CUDA and torch is installed :

nvcc --version

python -c "import torch; print(torch.cuda.is_available())"


This should return true. 


### 3. Install nvflare


pip install nvflare==2.4.0


Recommended: 

pip install numpy==1.22.0

pip install pandas==2.0.3


### 4. Set Required Environment Variables


export PYTHONPATH=$PYTHONPATH:[path to this dir]/app/code/

export NVFLARE_POC_WORKSPACE=[path to this dir]/poc-workspace/



### 5. Run nvflare simulator

nvflare simulator -c site1,site2 ./jobs/job








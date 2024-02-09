# Fast Surface Reconstruction for Point Clouds 
# MapMaker add-in

#### This work leverages the Neural Kernel Surface Reconstruction (CUDA) code developed by NVIDIA Corp. and the Authors cited below and blends it with our propietary MapMaker pipeline.

## Base Requirements

- Windows Subsystem for Linux (WSL) with Ubuntu 22.04<br>
- Linux user must be created as ```mapmaker```
- NVIDIA RTX20, RTX30 or RTX40 GPU with at least 8 Gb VRAM</ul>

## Installation (WIP)

Download Anaconda from https://www.anaconda.com/download#downloads and install, then type the commands listed below.

```shell
sudo apt install build-essential
conda create -n nksr python=3.10
conda activate nksr
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter pyntcloud plyfile pymeshlab pillow utm
export TORCH_VERSION=2.0.0
export CUDA_VERSION=cu118
pip install -U nksr -f https://nksr.huangjh.tech/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install python-pycg[full] -f https://pycg.s3.ap-northeast-1.amazonaws.com/packages/index.html
git clone https://github.com/cubantonystark/fsrpc_mapmaker.git
sudo nano .bashrc
```
#### place the following code at the end of .bashrc

```shell
#!/bin/bash
eval "(conda shell.bash hook)" > tmp.txt
source activate nksr
python pc2mesh_accel.py
```
<br>
## Citation

```bibtex
@inproceedings{huang2023nksr,
  title={Neural Kernel Surface Reconstruction},
  author={Huang, Jiahui and Gojcic, Zan and Atzmon, Matan and Litany, Or and Fidler, Sanja and Williams, Francis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4369--4379},
  year={2023}
}
```

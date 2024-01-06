#!/usr/bin/env bash

conda create -n fss python=3.8
source activate

conda activate fss
# Install PyTorch >=1.9 (see [PyTorch instructions](https://pytorch.org/get-started/locally/)).
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch

conda install -c pytorch faiss-gpu
pip install -r requirements.txt
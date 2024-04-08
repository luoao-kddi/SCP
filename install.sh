#!/bin/bash
conda create -n scp python=3.10 -y
conda activate scp
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt

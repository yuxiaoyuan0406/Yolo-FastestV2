#!/bin/bash
source /home/songwei/anaconda3/bin/activate
conda activate yolof >> /dev/null
python -c 'import torch;print(torch.utils.cmake_prefix_path)'

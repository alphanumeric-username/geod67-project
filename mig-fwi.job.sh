#!/bin/bash

module load anaconda3/2022.10

# source activate devito
source ~/.bashrc
#conda init bash

conda deactivate
conda activate geod67-unet-fwi
cd $CWD

# export DEVITO_LANGUAGE=openmp

python mig-fwi.py -l 0.001 -o ./data/vp_inv_it1500nbl40-prelu.bin -w ./data/it1500nbl40-prelu.weights.h5 # -c ./data/it1500nbl40.weights.h5
#!/bin/bash

# Define the array of GPU id and hyperparameters
gpu_id=0
lr_adj=(1e-3 1e-2 1e-1)
lr_feat=(1e-3 1e-2 1e-1)

# Activate the Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond


# Start grid search
for lr_a in "${lr_adj[@]}"
do
    for lr_f in "${lr_feat[@]}"
    do
        # Run training script
        python train_gcdm.py --dataset ppi-large --nlayers=2 --lr_feat=${lr_f} --lr_adj=${lr_a} --gpu_id=$((gpu_id % 8)) --r=0.001 --epochs=1500 \
        > hyperparam_optimLog/GCDM_PPI_Large/GCDM_PPI_Large_${lr_f}_${lr_a}.log 2>hyperparam_optimLog/GCDM_PPI_Large/GCDM_PPI_Large_${lr_f}_${lr_a}.err &
        
        # Increment GPU ID
        gpu_id=$((gpu_id + 1))
    done
done


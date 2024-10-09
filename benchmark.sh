#!/bin/bash

# Define the array of GPU ID and r values
gpu_id=0
r_values=(0.0001 0.0002 0.0004 0.0005 0.0010 0.0020)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond

for r in "${r_values[@]}"
do
    #if [ ${gpu_id} -eq 8 ] ; then
    #    break
    #fi

    nohup python train_sgdd.py --dataset yelp --nlayers=2 --lr_feat=1e-2 --gpu_id=$((gpu_id % 8)) --lr_adj=1e-2 --r=${r} > benchmarkLogs/Yelp_Results/SGDD_${r}.log 2>benchmarkLogs/Yelp_Results/SGDD_${r}.err &
    #nohup python train_coreset.py --dataset eg --r=${r} --method=random --gpu_id=$((gpu_id % 8)) > benchmarkLogs/EG_Results/CORE_RAND_${r}.log 2>benchmarkLogs/EG_Results/CORE_RAND_${r}.err &
    #nohup python train_gcond_transduct.py --dataset eg --nlayers=2 --lr_feat=1e-2 --gpu_id=$((gpu_id % 8))  --lr_adj=1e-2 --r=${r} >  benchmarkLogs/EG_Results/GCOND_${r}.log 2>benchmarkLogs/EG_Results/GCOND_${r}.err &

    gpu_id=$((gpu_id + 1))

    #if [ ${gpu_id} -eq 8 ] ; then
    #    break
    #fi

    #nohup python3 train_gcdm.py --dataset eg --nlayers=2 --lr_feat=1e-2 --gpu_id=$((gpu_id % 8)) --lr_adj=1e-2 --r=${r} > benchmarkLogs/EG_Results/GCDM_${r}.log 2>benchmarkLogs/EG_Results/GCDM_${r}.err &
    #nohup python train_coreset.py --dataset eg --r=${r} --method=herding --gpu_id=$((gpu_id % 8)) > benchmarkLogs/EG_Results/CORE_HERD_${r}.log 2>benchmarkLogs/EG_Results/CORE_HERD_${r}.err &
    #nohup python train_gcond_transduct.py --dataset hg --nlayers=2 --lr_feat=1e-2 --gpu_id=$((gpu_id % 8))  --lr_adj=1e-2 --r=${r} >  benchmarkLogs/HG_Results/GCOND_${r}.log 2>benchmarkLogs/HG_Results/GCOND_${r}.err &

    #gpu_id=$((gpu_id + 1))

    #if [ ${gpu_id} -eq 8 ] ; then
    #    break
    #fi

    #nohup python train_coreset.py --dataset eg --r=${r} --method=kcenter --gpu_id=$((gpu_id % 8)) > benchmarkLogs/EG_Results/CORE_KCNTR_${r}.log 2>benchmarkLogs/EG_Results/CORE_KCNTR_${r}.err &
    #gpu_id=$((gpu_id + 1))

done
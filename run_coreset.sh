#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond

declare -A datasets=(
  ["ppi"]=0.01
  ["ppi-large"]=0.001
  ["dblp"]=0.008
  ["pcg"]=0.04
  ["ogbn-pro"]=0.001
  ["yelp"]=0.0002
  # ["hg"]=0.04
  # ["eg"]=0.02
  # ["ppi-large"]=0.001

)

BASE_PATH="log/Coreset"

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=5
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_random.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
  python train_coreset.py  --gpu_id=$gpu_id --dataset $dataset --r=$r_value --method="random" >> $LOG_FILE_LABRAND0 2>/dev/null &

done

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=6
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_herding.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
  python train_coreset.py  --gpu_id=$gpu_id --dataset $dataset --r=$r_value --method="herding" >> $LOG_FILE_LABRAND0 2>/dev/null &

done

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=7
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_kcenter.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
  python train_coreset.py  --gpu_id=$gpu_id --dataset $dataset --r=$r_value --method="kcenter" >> $LOG_FILE_LABRAND0 2>/dev/null &

done
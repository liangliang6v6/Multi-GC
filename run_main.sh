#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond
# Declare datasets and corresponding r values
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

BASE_PATH="log/GCond"

# for dataset in "${!datasets[@]}"; do
#   r_value=${datasets[$dataset]}
#   gpu_id=0
#   LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_random_adj.log"

#   echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
#   python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=1e-2 --dataset $dataset --r=$r_value --subgraph=1 --method="random" >> $LOG_FILE_LABRAND0 2>/dev/null &

# done

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=1
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_random.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value, prob_rand=0)" > $LOG_FILE_LABRAND0
  python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=0 --dataset $dataset --r=$r_value --subgraph=1 --method="random" >> $LOG_FILE_LABRAND0 2>/dev/null &

done

# for dataset in "${!datasets[@]}"; do
#   r_value=${datasets[$dataset]}
#   gpu_id=2
#   LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_herding_adj.log"

#   echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
#   python train_sgdd.py  --gpu_id=$gpu_id --lr_adj=1e-2 --dataset $dataset --r=$r_value --subgraph=1 --method="herding" >> $LOG_FILE_LABRAND0 2>/dev/null &

# done

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=3
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_herding.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
  python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=0 --dataset $dataset --r=$r_value --subgraph=1 --method="herding" >> $LOG_FILE_LABRAND0 2>/dev/null &

done

# for dataset in "${!datasets[@]}"; do
#   r_value=${datasets[$dataset]}
#   gpu_id=4
#   LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_kcenter_adj.log"

#   echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
#   python train_sgdd.py  --gpu_id=$gpu_id --lr_adj=1e-2 --dataset $dataset --r=$r_value --subgraph=1 --method="kcenter" >> $LOG_FILE_LABRAND0 2>/dev/null &

# done

for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=5
  LOG_FILE_LABRAND0="${BASE_PATH}/${dataset}_r${r_value}_kcenter.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND0
  python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=0 --dataset $dataset --r=$r_value --subgraph=1 --method="kcenter" >> $LOG_FILE_LABRAND0 2>/dev/null &

done


for dataset in "${!datasets[@]}"; do
  r_value=${datasets[$dataset]}
  gpu_id=7
  LOG_FILE_LABRAND1="${BASE_PATH}/${dataset}_r${r_value}_prob.log"

  echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND1
  python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=0 --dataset $dataset --r=$r_value --lab_prob=1 >> $LOG_FILE_LABRAND1 2>/dev/null &
  
done

# for dataset in "${!datasets[@]}"; do
#   r_value=${datasets[$dataset]}
#   gpu_id=7
#   LOG_FILE_LABRAND1="log/set/${dataset}_r${r_value}_prob_adj.log"

#   echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value)" > $LOG_FILE_LABRAND1
#   python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=1e-2 --dataset $dataset --r=$r_value --lab_prob=1 >> $LOG_FILE_LABRAND1 2>/dev/null &
  
# done

# for dataset in "${!datasets[@]}"; do
#   r_value=${datasets[$dataset]}
#   gpu_id=4
#   LOG_FILE_LABRAND1="log/label/${dataset}_r${r_value}_prob.log"

#   echo "Running on GPU $gpu_id with dataset $dataset (r=$r_value, lab_rand=1)" > $LOG_FILE_LABRAND1
#   python train_gcond_transduct.py  --gpu_id=$gpu_id --lr_adj=1e-2 --dataset $dataset --r=$r_value --lab_rand=0 >> $LOG_FILE_LABRAND1 2>/dev/null &
  
# done


# Wait for all background processes to finish
wait



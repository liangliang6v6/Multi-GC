#!/bin/bash

# Activate the Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond

# Default values for the arguments
epochs=2000
reduction_rate=0.01
dataset="ppi"
lab_prob=0
subgraph=1
method="random"
loss="BCE"


# Parse command-line flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) epochs="$2"; shift ;;
        --reduction_rate) reduction_rate="$2"; shift ;;  # Single value
        --dataset) dataset="$2"; shift ;;  # Single value
        --method) method="$2"; shift ;;  # Single value
        --lab_prob) lab_prob="$2"; shift ;;  # Single value
        --subgraph) subgraph="$2"; shift ;;  # Single value
        --loss) loss="$2"; shift ;;  # Single value
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Fixed hyperparameter values (you can modify as needed)
hidden=(64 128)
lr_adj=(1e-1 1e-2)
lr_feat=(1e-1 1e-2)
lr_model=(1e-1 1e-2)
weight_decay=(0 5e-4)
dropout=(0 0.05)


gpu_ids=(0 1 2 3 4 5 6 7)  # 8 GPUs available

# Create a counter to assign GPUs
gpu_counter=0

# Loop over the combinations of learning rate, hidden size, nlayers, weight decay, dropout, and alpha
for h in "${hidden[@]}"; do
  for lr_a in "${lr_adj[@]}"; do
    for lr_f in "${lr_feat[@]}"; do
      for lr_m in "${lr_model[@]}"; do
        for wd in "${weight_decay[@]}"; do
          for d in "${dropout[@]}"; do
            # Assign the GPU ID in round-robin fashion
            gpu_id=${gpu_ids[$gpu_counter]}

            # Construct the command
            cmd="python train_gcond_transduct.py --dataset $dataset --method $method --lr_adj $lr_a --lr_feat $lr_f --lr_model $lr_m --hidden $h --weight_decay $wd --dropout $d --gpu_id $gpu_id --epochs $epochs --lab_prob $lab_prob --subgraph $subgraph --loss $loss"

            # Define log directory and log file path
            #log_dir="log/GCOND_log/${dataset}/reduction${reduction_rate}/${method}"
            log_dir="log/GCOND_log/softmargin/withadj/${dataset}/reduction${reduction_rate}/${method}"
            log_file="${log_dir}/_h${h}_lr_adj${lr_a}_lr_feat${lr_f}_lr_model${lr_m}_wd${wd}_d${d}.log"
            err_file="${log_dir}/_h${h}_lr_adj${lr_a}_lr_feat${lr_f}_lr_model${lr_m}_wd${wd}_d${d}.err"

            # Create the log directory if it does not exist
            mkdir -p "$log_dir"

            # Run the command in the background and redirect output to the log file
            echo "Running on GPU $gpu_id: $cmd"
            $cmd > "$log_file" 2> "$err_file" &

            # Update GPU counter
            gpu_counter=$(( (gpu_counter + 1) % 8 ))

            # Wait for all background jobs to finish if all GPUs are occupied
            if [ $gpu_counter -eq 0 ]; then
                wait
            fi
          done
        done
      done
    done
  done
done

# Wait for any remaining jobs
wait

echo "All grid search runs complete!"

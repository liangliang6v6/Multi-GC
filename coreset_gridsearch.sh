#!/bin/bash

# Activate the Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GCond

# Default values for the arguments
epochs=400
reduction_rate=0.01
dataset="ppi"
method="random"

# Parse command-line flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) epochs="$2"; shift ;;
        --reduction_rate) reduction_rate="$2"; shift ;;  # Single value
        --dataset) dataset="$2"; shift ;;  # Single value
        --method) method="$2"; shift ;;  # Single value
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Fixed hyperparameter values (you can modify as needed)
lr_values=(0.01 0.001 0.0001)
hidden_values=(128 256 512)
nlayers_values=(2 3 4)  # New: Number of layers
weight_decay_values=(0 5e-4 1e-3 5e-3)  # New: Weight decay
gpu_ids=(0 1 2 3 4 5 6 7)  # 8 GPUs available

# Create a counter to assign GPUs
gpu_counter=0

# Loop over the combinations of learning rate, hidden size, nlayers, weight decay, dropout, and alpha
for lr in "${lr_values[@]}"; do
  for hidden in "${hidden_values[@]}"; do
    for nlayers in "${nlayers_values[@]}"; do
      for weight_decay in "${weight_decay_values[@]}"; do
            # Assign the GPU ID in round-robin fashion
            gpu_id=${gpu_ids[$gpu_counter]}

            # Construct the command
            cmd="python train_coreset.py --dataset $dataset --method $method --lr $lr --hidden $hidden --reduction_rate $reduction_rate --nlayers $nlayers --weight_decay $weight_decay --gpu_id $gpu_id --epochs $epochs --seed 846"

            # Define log directory and log file path
            log_dir="coreset_logs/${dataset}/${method}/reduction${reduction_rate}"
            log_file="${log_dir}/_lr${lr}_hidden${hidden}_nlayers${nlayers}_weight_decay${weight_decay}.log"

            # Create the log directory if it does not exist
            mkdir -p "$log_dir"

            # Run the command in the background and redirect output to the log file
            echo "Running on GPU $gpu_id: $cmd"
            $cmd > "$log_file" 2>&1 &

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

# Wait for any remaining jobs
wait

echo "All grid search runs complete!"

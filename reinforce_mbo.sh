#!/bin/bash
unset WANDB_RUN_ID
export WANDB_API_KEY=0e54236b9b51e012ca75d44fff1569c424ddc9da
export WANDB_PROJECT=TACO
export WANDB_DIR="~/wandb_cache"

tasks=("k562")
seeds=(0 1 2 3 4)
# tfbs_lambdas=(0 0.1 0.5 1.0 5 10)

# Define the tfbs_lambda values to iterate over
tfbs_lambdas=(0.0)
lr=1e-4

pkill -f oc_gpu

# Iterate over each task and tfbs_lambda combination
for task in "${tasks[@]}"; do
    for tfbs_lambda in "${tfbs_lambdas[@]}"; do
        # Launch processes for each seed in parallel
        # Each seed runs on the corresponding device number
        for seed in "${seeds[@]}"; do
            # Use the seed value as the device number
            device=$seed
            
            echo "Starting task $task with seed $seed on device $device"
            
            # Launch the Python script in the background to run in parallel
            CUDA_VISIBLE_DEVICES=$device \
                python reinforce_mbo.py \
                --task "$task" \
                --seed "$seed" \
                --lr "$lr" \
                --tfbs_lambda "$tfbs_lambda" \
                --wandb_log &  # Run in background
                
            # Optional: Add a small delay to prevent potential race conditions
            sleep 1
        done
        
        # Wait for all processes with different seeds to complete before moving to the next parameter
        wait
    done
done

echo "All processes completed"
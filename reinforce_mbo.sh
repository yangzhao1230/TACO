#!/bin/bash
unset WANDB_RUN_ID
export WANDB_API_KEY=0e54236b9b51e012ca75d44fff1569c424ddc9da
export WANDB_PROJECT=TACO
export WANDB_DIR="~/wandb_cache"

device=0
tasks=("k562")
seeds=(4)
# seeds=(4)
# tfbs_lambdas=(0 0.1 0.5 1.0 5 10)  # Define the tfbs_lambda values to iterate over
tfbs_lambdas=(0.0)
lr=1e-4

pkill -f oc_gpu
# Iterate over each parameter combination and call the Python script
for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        for tfbs_lambda in "${tfbs_lambdas[@]}"; do
            # Call the Python script and pass all parameters
            CUDA_VISIBLE_DEVICES=$device \
                python reinforce_mbo.py \
                --task "$task" \
                --seed "$seed" \
                --lr "$lr" \
                --tfbs_lambda "$tfbs_lambda" \
                --wandb_log  # Uncomment if wandb logging is needed
        done
    done
done

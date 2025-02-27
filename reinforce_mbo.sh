# #!/bin/bash
# unset WANDB_RUN_ID
# device=3

# # Define parameter lists
# # tasks=('hepg2' 'k562')
# tasks=("hepg2")
# levels=('mbo')
# seeds=(0 1 2 3 4)
# lambdas=(0.0)  # Define lambda values to iterate over

# # Iterate over each parameter combination and call the Python script
# for task in "${tasks[@]}"; do
#     for level in "${levels[@]}"; do
#         for seed in "${seeds[@]}"; do
#             for tfbs_lambda in "${lambdas[@]}"; do
#                 # Call the Python script and pass all parameters
#                 CUDA_VISIBLE_DEVICES=$device \
#                     python debug.py \
#                     --task "$task" --level "$level" --seed "$seed"  \
#                     --tfbs_lambda "$tfbs_lambda" \
#                     --lr 0.0001 \
#                     --save_dir results_2.5.8 \
#                     # wo ren--wandb_log
#             done
#         done
#     done
# done
#!/bin/bash
unset WANDB_RUN_ID

tasks=("hepg2" "sknsh")   
levels=('mbo')             
seeds=(0 1 2 3 4)        
lambdas=(0.01)            
save_dir="results_2.5.5"   
lr=0.0001                  


run_experiment() {
    local seed=$1
    local device=$seed  

    echo "Starting experiments for seed=$seed on GPU=$device"

    for task in "${tasks[@]}"; do
        for level in "${levels[@]}"; do
            for tfbs_lambda in "${lambdas[@]}"; do
                CUDA_VISIBLE_DEVICES=$device \
                    python reinforce_mbo.py \
                    --task "$task" --level "$level" --seed "$seed" \
                    --tfbs_lambda "$tfbs_lambda" \
                    --lr "$lr" \
                    --save_dir "$save_dir"

                sleep 2  
            done
        done
    done

    echo "Finished experiments for seed=$seed on GPU=$device"
}

for seed in "${seeds[@]}"; do
    run_experiment "$seed" &  #
done

wait

echo "All experiments finished!"

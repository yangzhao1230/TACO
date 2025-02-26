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

# 定义参数
tasks=("hepg2")            # 任务列表
levels=('mbo')             # 任务级别
seeds=(0 1 2 3 4)          # 每个 seed 绑定到不同的 GPU
lambdas=(0.0)              # lambda 值
save_dir="results_2.5.8"   # 结果存储目录
lr=0.0001                  # 学习率

# 定义执行函数（每个 seed 绑定到一个 GPU，并且任务内部串行执行）
run_experiment() {
    local seed=$1
    local device=$seed  # 让 seed i 绑定到 GPU i

    echo "Starting experiments for seed=$seed on GPU=$device"

    for task in "${tasks[@]}"; do
        for level in "${levels[@]}"; do
            for tfbs_lambda in "${lambdas[@]}"; do
                CUDA_VISIBLE_DEVICES=$device \
                    python debug.py \
                    --task "$task" --level "$level" --seed "$seed" \
                    --tfbs_lambda "$tfbs_lambda" \
                    --lr "$lr" \
                    --save_dir "$save_dir"

                # 可选：如果需要延迟防止 GPU 负载过高
                sleep 2  
            done
        done
    done

    echo "Finished experiments for seed=$seed on GPU=$device"
}

# 遍历所有 seed，每个 seed 绑定到不同的 GPU，并行运行
for seed in "${seeds[@]}"; do
    run_experiment "$seed" &  # 后台运行每个 seed 绑定的任务
done

# 等待所有任务完成
wait

echo "All experiments finished!"

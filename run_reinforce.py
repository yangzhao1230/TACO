import random
import argparse

import numpy as np
import pandas as pd
import torch

from dna_optimizers.base_optimizer import get_fitness_info
from dna_optimizers.tfbs_optimizer import tfbs_mbo_optimizer

def get_meme_and_ppms_path(task):
    if task == "complex" or task == "defined":
        meme_path = "/blob/ICLR/tfbs/yeast/20240906120455_JASPAR2024_combined_matrices_1185532_meme.txt"
        ppms_path = "/blob/ICLR/tfbs/yeast/selected_ppms.csv"
    elif task == "hepg2" or task == "k562" or task == "sknsh":
        meme_path = "/blob/ICLR/tfbs/human/20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt"
        ppms_path = "/blob/ICLR/tfbs/human/selected_ppms.csv"
    else:
        raise ValueError(f"Task {task} not supported.")
    return meme_path, ppms_path
    
def get_model_name_or_path(cell):
    if cell == "complex" or cell == "defined":
        return "/blob/ICLR/reglm/yeast_reglm.ckpt"
    elif cell == "hepg2" or cell == "k562" or cell == "sknsh":
        return "/blob/ICLR/reglm/human_reglm.ckpt"

def get_prefix_label(task, level):
    """
    prefix_label come from reglm which to control the condition, w
    """
    return ""
    

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for OptimizerArguments.")

    parser.add_argument("--task", type=str, default="hepg2", help="The task name.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type (e.g., 'cuda' or 'cpu').")

    parser.add_argument("--max_oracle_calls", type=int, default=4000000, help="Maximum number of oracle calls.")
    parser.add_argument("--max_strings", type=int, default=384000000, help="Maximum number of strings.")
    
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--env_log_interval", type=int, default=256, help="Environment logging interval.")

    parser.add_argument("--wandb_log", action='store_true', help="Enable logging with wandb.")
    parser.add_argument("--train_log_interval", type=int, default=4, help="Training log interval.")

    parser.add_argument("--level", type=str, default="mbo", help="Difficulty level.")
    parser.add_argument("--e_size", type=int, default=100, help="Experience size.")
    parser.add_argument("--e_batch_size", type=int, default=24, help="Experience batch size.")
    parser.add_argument("--priority", type=bool, default=True, help="Use priority in experience replay.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tfbs_lambda", type=float, default=1, help="Lambda value for TFBS.")

    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def main():

    optimizer_args = parse_args()
    get_model_name_or_path
    
    optimizer_args.prefix_label = get_prefix_label(optimizer_args.task, optimizer_args.level)
    print(f"Prefix Label: {optimizer_args.prefix_label}")
    optimizer_args.model_name_or_path = get_model_name_or_path(optimizer_args.task)
    
    optimizer_args.max_len, min_fitness, max_fitness = get_fitness_info(optimizer_args.task)
    
    optimizer_args.meme_path, optimizer_args.ppms_path = get_meme_and_ppms_path(optimizer_args.task)

    optimizer_args.tfbs_reward_path = f"reward_dict/results_{optimizer_args.task}_mbo.json"

    optimizer_args.wandb_run_name = (
        "mbo_"
        f"{optimizer_args.task}_{optimizer_args.level}_"
        f"e_size_{optimizer_args.e_size}_"
        f"e_bs_{optimizer_args.e_batch_size}_"
        f"lr_{optimizer_args.lr}_"
        f"lambda_{optimizer_args.tfbs_lambda}_"
        f"seed_{optimizer_args.seed}"
    )
    set_seed(optimizer_args.seed)
    print(optimizer_args)
    # import debugpy; debugpy.connect(5678); debugpy.wait_for_client(); debugpy.breakpoint()
    optimizer = tfbs_mbo_optimizer(optimizer_args)
    
    init_data_path = f"./data/{optimizer_args.task}/{optimizer_args.level}.csv"
    starting_sequences = pd.read_csv(init_data_path)
    
    starting_sequences['target'] = (starting_sequences['target'] - min_fitness)/(max_fitness - min_fitness)
    starting_sequences = starting_sequences.sort_values('target', ascending=False).head(128)
        
    optimizer.optimize(optimizer_args, starting_sequences)

if __name__ == "__main__":
    main()
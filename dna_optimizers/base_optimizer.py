import os
import tdc
import itertools
import time
import yaml
import wandb
import torch
import numpy as np

import reglm_src.reglm.dataset, reglm_src.reglm.lightning, reglm_src.reglm.utils, reglm_src.reglm.metrics
from .experience import Experience

def evaluate(round_df, starting_sequences):
    data = round_df.sort_values(by='true_score', ascending=False).iloc[:128]
    
    top_fitness = data.iloc[:16]['true_score'].mean().item()
    median_fitness = data['true_score'].median().item()
    
    seqs = data['sequence'].tolist()
    
    distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
    diversity = np.median(distances) if distances else 0.0
    
    inits = starting_sequences['sequence'].tolist()
    novelty_distances = [min(distance(seq, init_seq) for init_seq in inits) for seq in seqs]
    novelty = np.median(novelty_distances) if novelty_distances else 0.0
    
    return {
        'top': top_fitness,
        'fitness': median_fitness,
        'diversity': diversity,
        'novelty': novelty
    }
    
def evaluate_with_oracle(round_df, starting_sequences):
    data = round_df.sort_values(by='true_score', ascending=False).iloc[:128]
    
    top_fitness = data.iloc[:16]['true_score'].mean().item()
    median_fitness = data['true_score'].median().item()
    
    top_fitness_oracle = data.iloc[:16]['oracle_score'].mean().item()
    median_fitness_oracle = data['oracle_score'].median().item()
    
    seqs = data['sequence'].tolist()
    
    distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
    diversity = np.median(distances) if distances else 0.0
    
    inits = starting_sequences['sequence'].tolist()
    novelty_distances = [min(distance(seq, init_seq) for init_seq in inits) for seq in seqs]
    novelty = np.median(novelty_distances) if novelty_distances else 0.0
    
    return {
        'top': top_fitness,
        'fitness': median_fitness,
        'diversity': diversity,
        'novelty': novelty,
        'top_oracle': top_fitness_oracle,
        'fitness_oracle': median_fitness_oracle
    }
    
def get_fitness_info(cell):
    if cell == 'complex':
        length = 80
        min_fitness = 0
        max_fitness = 17
    elif cell == 'defined':
        length = 80
        min_fitness = 0
        max_fitness = 17
    elif cell == 'hepg2':
        length = 200
        min_fitness = -6.051336
        max_fitness = 10.992575
    elif cell == 'k562':
        length = 200
        min_fitness = -5.857445
        max_fitness = 10.781755
    elif cell == 'sknsh':
        length = 200
        min_fitness = -7.283977
        max_fitness = 12.888308
    else:
        raise NotImplementedError()
    return length, min_fitness, max_fitness
    
def distance(s1, s2):
    return sum([1 if i != j else 0 for i, j in zip(list(s1), list(s2))])

def diversity(seqs):
    divs = []
    for s1, s2 in itertools.combinations(seqs, 2):
        divs.append(distance(s1, s2))
    return sum(divs) / len(divs)

def mean_distance(seq, seqs):
    divs = []
    for s in seqs:
        divs.append(distance(seq, s))
    return sum(divs) / len(divs)

class BaseOptimizer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        
        self.task = cfg.task
        _, self.min_fitness, self.max_fitness = get_fitness_info(self.task)
        self.assign_target(cfg)

        self.max_oracle_calls = cfg.max_oracle_calls
        self.env_log_interval = cfg.env_log_interval
        
        self.dna_buffer = dict()
        self.mean_score = 0

        self.last_log = 0
        self.last_log_time = time.time()
        self.last_logging_time = time.time()
        self.total_count = 0
        self.invalid_count = 0
        self.redundant_count = 0
        
        if cfg.wandb_log:
            wandb.init(
                project="TACO",
                name=cfg.wandb_run_name,
            )

        self.device = torch.device(cfg.device)
       
        self.experience = Experience(cfg.e_size, cfg.priority)

    @property
    def budget(self):
        return self.max_oracle_calls
    
    @property
    def finish(self):
        return len(self.dna_buffer) >= self.max_oracle_calls
    
    def assign_target(self, cfg): 
        if self.task == "complex":
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/yeast_regression_paired_complex_offline.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == "defined":
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/yeast_regression_paired_defined_offline.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
            
        elif self.task == 'hepg2':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_hepg2_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == 'k562':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_k562_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
            
        elif self.task == 'sknsh':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_sknsh_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        else:
            raise NotImplementedError
        
        self.target.eval()

    def assign_oracle(self, cfg): 
        # Check if task is yeast promoter
        if self.task == "complex":
            # Load complex_oracle from checkpoint
            self.oracle = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/yeast_regression_paired_complex.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == "defined":
            self.oracle = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/yeast_regression_paired_defined.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
            
        elif self.task == 'hepg2':
            self.oracle = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_hepg2.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == 'k562':
            self.oracle = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_k562.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
            
        elif self.task == 'sknsh':
            self.oracle = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/ICLR/oracle/human_regression_paired_sknsh.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        else:
            raise NotImplementedError
        
        self.oracle.eval()


    def normalize_target(self, score):
        return (score - self.min_fitness) / (self.max_fitness - self.min_fitness)
    
    @torch.no_grad()
    def score_enformer(self, dna):
        if len(self.dna_buffer) > self.max_oracle_calls:
            return 0
        
        score = self.target([dna]).squeeze(0).item()
        score = self.normalize_target(score)
        # score = self.oracle([dna]).squeeze(0).item()
        if dna in self.dna_buffer:
            self.dna_buffer[dna][2] += 1
            self.redundant_count += 1

        else:
            self.dna_buffer[dna] = [float(score), len(self.dna_buffer)+1, 1]

        return self.dna_buffer[dna][0]
    
    def add_dna(self, dna, score):
        if dna in self.dna_buffer:
            self.dna_buffer[dna][2] += 1
            self.redundant_count += 1
        else:
            self.dna_buffer[dna] = [float(score), len(self.dna_buffer)+1, 1]

    def predict_enformer(self, dna_list):
        st = time.time()
        assert type(dna_list) == list
        self.total_count += len(dna_list)
        score_list = []
        for dna in dna_list:
            score_list.append(self.score_enformer(dna))
                
            if len(self.dna_buffer) % self.env_log_interval == 0 and len(self.dna_buffer) > self.last_log:
                self.sort_buffer()
                # self.log_intermediate()
                self.last_log_time = time.time()
                self.last_log = len(self.dna_buffer)

        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list

    def optimize(self, cfg):
        raise NotImplementedError

    def sort_buffer(self):
        self.dna_buffer = dict(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
            

    def __len__(self):
        return len(self.dna_buffer)
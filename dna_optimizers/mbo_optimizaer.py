import itertools
import json
import time
import os

import numpy as np
import torch
from torch import optim
import wandb
import pandas as pd

from .experience import Experience
import reglm_src.reglm.dataset
import reglm_src.reglm.lightning
import reglm_src.reglm.metrics
import reglm_src.reglm.utils
import scripts.motifs
import scripts.utils


def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

def distance(s1, s2):
    return sum([1 if i != j else 0 for i, j in zip(list(s1), list(s2))])

def diversity(seqs):
    divs = []
    for s1, s2 in itertools.combinations(seqs, 2):
        divs.append(distance(s1, s2))
    return sum(divs) / len(divs)

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
    
    ## surrogate
    top_fitness = data.iloc[:16]['true_score'].mean().item()
    median_fitness = data['true_score'].median().item()
    
    ## oracle
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

class mbo_optimizer():

    def __init__(self, cfg):

        # base init
        self.cfg = cfg
        self.task = cfg.task
        _, self.min_fitness, self.max_fitness = get_fitness_info(self.task)
        self.assign_target(cfg) # assign target model for guided search
        self.assign_oracle(cfg) # assign oracle model for final evaluation

        self.dna_buffer = dict()

        if cfg.wandb_log:
            wandb.init(
                project="ICLR_Rebuttal_MBO_60",
                name=cfg.wandb_run_name,
            )

        self.device = torch.device(cfg.device)
        self.experience = Experience(cfg.e_size, cfg.priority)

        # reinforce init
        self.prefix_label =  cfg.prefix_label
        if cfg.model_name_or_path is not None:
            self.agent = reglm_src.reglm.lightning.LightningModel.load_from_checkpoint(
                cfg.model_name_or_path
            )     
        else:
            self.agent = reglm_src.reglm.lightning.LightningModel()
        self.agent.to(self.device)
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg.lr)
        self.vocab = list(self.agent.label_stoi.keys()) + list(self.agent.base_stoi.keys())
        self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        self.predict = self.predict_enformer
        # tfbs init
        self.tfbs_lambda = cfg.tfbs_lambda
        motifs, bg = scripts.motifs.read_meme(
            cfg.meme_path
        )
        self.tfbs_reward_dict = json.load(open(cfg.tfbs_reward_path, 'r'))
        sel = scripts.utils.load_csv(
            cfg.ppms_path
        ).Matrix_id.tolist()
        self.motif2idx = {name: i for i, name in enumerate(sel)}
        self.idx2motif = {i: name for i, name in enumerate(sel)}
        motifs = [m for m in motifs if m.name.decode() in sel]
        self.motifs = motifs
        self.bg = bg

    def assign_target(self, cfg): 
        if self.task == "complex":
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/Data/DNA/Design/guide_model/yeast_regression_paired_complex_offline.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == "defined":
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                "/blob/Data/DNA/Design/guide_model/yeast_regression_paired_defined_offline.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == 'hepg2':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                # "/blob/Data/DNA/Design/guide_model/human_regression_paired_hepg2_mbo_60.ckpt",
                "/blob/Data/DNA/Design/guide_model/human_regression_paired_hepg2_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        elif self.task == 'k562':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                # "/blob/Data/DNA/Design/guide_model/human_regression_paired_k562_mbo_60.ckpt",
                "/blob/Data/DNA/Design/guide_model/human_regression_paired_k562_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
            
        elif self.task == 'sknsh':
            self.target = reglm_src.reglm.regression.EnformerModel.load_from_checkpoint(
                # "/blob/Data/DNA/Design/guide_model/human_regression_paired_sknsh_mbo_60.ckpt",
                "/blob/Data/DNA/Design/guide_model/human_regression_paired_sknsh_mbo.ckpt",
                map_location=cfg.device,
            ).to(cfg.device)
        else:
            raise NotImplementedError
        
        self.target.eval()

    def assign_oracle(self, cfg): 
        if self.task == "complex":
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

    def sort_buffer(self):
        self.dna_buffer = dict(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def predict_enformer(self, dna_list):
        st = time.time()
        assert type(dna_list) == list
        score_list = []
        for dna in dna_list:
            score_list.append(self.score_enformer(dna))
            self.sort_buffer()

        return score_list

    def normalize_target(self, score):
        return (score - self.min_fitness) / (self.max_fitness - self.min_fitness)
    
    @torch.no_grad()
    def score_enformer(self, dna):
    
        score = self.target([dna]).squeeze(0).item()
        score = self.normalize_target(score)
        # score = self.oracle([dna]).squeeze(0).item()
        if dna in self.dna_buffer:
            self.dna_buffer[dna][2] += 1

        else:
            self.dna_buffer[dna] = [float(score), len(self.dna_buffer)+1, 1]

        return self.dna_buffer[dna][0]

    def update(self, obs, rewards, nonterms, episode_lens, cfg, metrics, log):
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms)

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        
        loss_p = self.alpha * ( - (1 / logprobs.sum(0, keepdim=True)).mean())
        loss = loss_pg + loss_p

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

    def optimize(self, cfg, starting_sequences):
        # import pdb; pdb.set_trace()
        # results' dataframe
        columns = ['round', 'sequence', 'true_score' , 'oracle_score']
        df = pd.DataFrame(columns=columns)
        summary_columns = ["round", "top", "fitness", "diversity", "novelty", "top_oracle", "fitness_oracle"]
        summary_df = pd.DataFrame(columns=summary_columns)

        # 1 add inital experience
        init_seqs = starting_sequences['sequence'].tolist()
        init_tokens = self.agent.encode(starting_sequences['sequence'].tolist(), [self.prefix_label] * len(init_seqs), add_start=True)
        init_tokens = init_tokens.T.to(self.device)
        # 
        init_rewards = torch.zeros(cfg.max_len+len(self.prefix_label), len(init_seqs))
        init_scores = starting_sequences['target'].tolist()
        init_rewards[-1, :] = torch.tensor(init_scores, dtype=torch.float32)
        init_rewards = init_rewards.to(self.device)
        # init_scores = torch.tensor(init_scores, dtype=torch.float32).to(self.device)
        init_nonterms = [False] * len(self.prefix_label) + [True] * (cfg.max_len) + [False] 
        init_nonterms = torch.tensor(init_nonterms, dtype=torch.bool)
        init_nonterms = init_nonterms.unsqueeze(-1).expand(-1, len(init_seqs)).to(self.device)
        init_lens = [cfg.max_len] * len(init_seqs)
        init_lens = torch.tensor(init_lens, dtype=torch.long)
        self.experience.add_experience(init_seqs, init_tokens, init_scores, init_rewards, init_nonterms, init_lens)
        
        for i in range(cfg.epoch):
            self.update(init_tokens, init_rewards, init_nonterms, init_lens, cfg, dict(), False)
        
        metrics = dict() 
        print('Start training ... ')

        for it in range(1, cfg.max_iter + 1):

            with torch.no_grad():
                labels = [self.prefix_label] * cfg.batch_size
                obs, rewards, nonterms, episode_lens = self.agent.get_data(labels, cfg.max_len)

            dna_list = []
            for dna in obs.cpu().numpy().T:
                dna_seq = self.agent.decode(dna, ignore_num=len(self.prefix_label)+1)[0]
                assert len(dna_seq) == cfg.max_len
                dna_list.append(dna_seq)
                
            # score produced by the surrogate model
            score = np.array(self.predict(dna_list)) # pay attention to the predict function
            scores = torch.tensor(score, dtype=torch.float32, device=self.device)

            # score produced by the oracle model, not used in training, just for monitoring
            with torch.no_grad():
                oracle_score = self.oracle(dna_list).squeeze(-1).detach().cpu().numpy()
                oracle_score =[self.normalize_target(s) for s in oracle_score]
                oracle_score = np.array(oracle_score)
    
             
            if self.tfbs_lambda > 0:
                # ********* TFBS **********
                tfbs_sites = scripts.motifs.scan(dna_list, self.motifs, self.bg)
                tfbs_sites["start"] -= 1 # 1-based to 0-based
                tfbs_struc = scripts.motifs.generate_tfbs_structure(tfbs_sites)
                for idx, tfbs_info in enumerate(tfbs_struc):
                    seq_id = list(tfbs_info.keys())[0]
                    tfbs_list = tfbs_info[seq_id]
                    for tfbs in tfbs_list:
                        start, end = tfbs['start'], tfbs['end']
                        region_reward = self.tfbs_reward_dict[tfbs['Matrix_id']] * self.tfbs_lambda
                        rewards[end+len(self.prefix_label)-1, idx] += region_reward # end+2-1 because of 1-based indexing


            log = False
            rewards[-1, :] += scores
            
            if len(self.experience) > cfg.e_batch_size:
                e_obs, e_scores, e_rewards, e_nonterms, e_episode_lens = self.experience.sample(cfg.e_batch_size, self.device)
                e_L, e_B = e_obs.shape
                L, B = obs.shape

                f_L = max(e_L, L)

                f_obs = torch.zeros((f_L, cfg.batch_size + cfg.e_batch_size), dtype=torch.long, device=self.device)
                f_nonterms = torch.zeros((f_L, cfg.batch_size + cfg.e_batch_size), dtype=torch.bool, device=self.device)

                f_obs[:L, :B] = obs
                f_obs[:e_L, B:] = e_obs

                f_nonterms[:L, :B] = nonterms
                f_nonterms[:e_L, B:] = e_nonterms
                
                # f_scores = torch.cat([scores, e_scores], dim=-1)
                f_rewards = torch.cat([rewards, e_rewards], dim=-1)
                f_episode_lens = torch.cat([episode_lens, e_episode_lens])
               
                for i in range(cfg.epoch):
                    self.update(f_obs, f_rewards, f_nonterms, f_episode_lens, cfg, metrics, log)
            else:
                for i in range(cfg.epoch):
                    # import pdb; pdb.set_trace()
                    self.update(obs, rewards, nonterms, episode_lens, cfg, metrics, log)

            self.experience.add_experience(dna_list, obs, score, rewards, nonterms, episode_lens)

            round_df = pd.DataFrame({'round': [it]*len(dna_list), 'sequence': dna_list, 'true_score': score, 'oracle_score': oracle_score})
            df = pd.concat([df, round_df], ignore_index=True)
            
            round_results = evaluate_with_oracle(round_df, starting_sequences)

            round_results['round'] = it


            round_results_df = pd.DataFrame([round_results])
            summary_df = pd.concat([summary_df, round_results_df], ignore_index=True)
            
            if cfg.wandb_log:
                wandb.log(round_results)
            print(f"Round {it} finished")
            print(round_results)
            
        # os.makedirs('results_serial', exist_ok=True)
        # cfg.save_dir
        os.makedirs(cfg.save_dir, exist_ok=True)
        save_name = f'{cfg.save_dir}/mbo_{cfg.tfbs_lambda}_{cfg.lr}_{cfg.task}_{cfg.level}_{cfg.seed}.csv'
        # save_name = f'results_serial/mbo_{cfg.tfbs_lambda}_{cfg.lr}_{cfg.task}_{cfg.level}_{cfg.seed}.csv'
        df.to_csv(save_name, index=False)
        # summary_name = f'results_serial/mbo_{cfg.tfbs_lambda}_{cfg.lr}_{cfg.task}_{cfg.level}_{cfg.seed}_summary.csv'
        summary_name = f'{cfg.save_dir}/mbo_{cfg.tfbs_lambda}_{cfg.lr}_{cfg.task}_{cfg.level}_{cfg.seed}_summary.csv'
        summary_df.to_csv(summary_name, index=False)

        print('max training string hit')
        wandb.finish()
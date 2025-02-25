import os
import sys
import wandb
import torch
from torch import optim
import numpy as np
import json
import pandas as pd
import reglm_src.reglm.regression

from .reinforce_optimizer import reinforce_optimizer
import reglm_src.reglm.dataset, reglm_src.reglm.lightning, reglm_src.reglm.utils, reglm_src.reglm.metrics
import scripts.utils, scripts.motifs

from dna_optimizers.base_optimizer import evaluate_with_oracle

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)
    
class tfbs_mbo_optimizer(reinforce_optimizer):
    
    def _init(self, cfg):
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
        self.access_oracle = self.oracle_enformer
        
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # self.agent = src.reglm.lightning.LightningModel()
        # self.agent.to(self.device)
        # self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg.lr)    
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
        
        self.assign_oracle(cfg)

    @torch.no_grad()
    def oracle_enformer(self, dna):
        score = self.oracle([dna]).squeeze(0).item()
        score = self.normalize_target(score)
        return score

    def optimize(self, cfg, starting_sequences):

        columns = ['round', 'sequence', 'true_score']
        df = pd.DataFrame(columns=columns)
        summary_columns = ["round", "top", "fitness", "diversity", "novelty"]
        summary_df = pd.DataFrame(columns=summary_columns)

        init_seqs = starting_sequences['sequence'].tolist()
        init_tokens = self.agent.encode(starting_sequences['sequence'].tolist(), [self.prefix_label] * len(init_seqs), add_start=True)
        init_tokens = init_tokens.T.to(self.device)
        
        init_rewards = torch.zeros(cfg.max_len+len(self.prefix_label), len(init_seqs))
        init_scores = starting_sequences['target'].tolist()
        init_rewards[-1, :] = torch.tensor(init_scores, dtype=torch.float32)
        init_rewards = init_rewards.to(self.device)

        init_nonterms = [False] * len(self.prefix_label) + [True] * (cfg.max_len) + [False] 
        init_nonterms = torch.tensor(init_nonterms, dtype=torch.bool)
        init_nonterms = init_nonterms.unsqueeze(-1).expand(-1, len(init_seqs)).to(self.device)
        init_lens = [cfg.max_len] * len(init_seqs)
        init_lens = torch.tensor(init_lens, dtype=torch.long)
        self.experience.add_experience(init_seqs, init_tokens, init_scores, init_rewards, init_nonterms, init_lens)
        
        for i in range(cfg.epoch):
            self.update(init_tokens, init_rewards, init_nonterms, init_lens, cfg, dict(), False)
        
        train_steps = 0
        eval_strings = 0
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
                
            score = np.array(self.predict(dna_list))
            # oracle_score = np.array(self.oracle(dna_list))
            with torch.no_grad():
                oracle_score = self.oracle(dna_list).squeeze(-1).detach().cpu().numpy()
                oracle_score =[self.normalize_target(s) for s in oracle_score]
                oracle_score = np.array(oracle_score)
                
            scores = torch.tensor(score, dtype=torch.float32, device=self.device)

            # one of our key contributions is the TFBS reward
            if self.tfbs_lambda > 0:
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

            train_steps += 1

            log = False

            rewards[-1, :] += scores
            
            # replay buffer
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

                f_rewards = torch.cat([rewards, e_rewards], dim=-1)
                f_episode_lens = torch.cat([episode_lens, e_episode_lens])
               
                for i in range(cfg.epoch):
                    self.update(f_obs, f_rewards, f_nonterms, f_episode_lens, cfg, metrics, log)
            else:
                for i in range(cfg.epoch):
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

        os.makedirs('results_mbo', exist_ok=True)
        save_name = f'results_mbo/{cfg.task}_{cfg.level}_{cfg.tfbs_lambda}_{cfg.seed}.csv'
        df.to_csv(save_name, index=False)
        # summary_name = f'results_mbo/{cfg.task}_{cfg.level}_{cfg.tfbs_lambda}_{cfg.seed}_summary.csv'
        # summary_df.to_csv(summary_name, index=False)
        print('max training string hit')
        wandb.finish()
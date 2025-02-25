import os
import sys
import wandb
import hydra
import torch
from torch import optim
from torch import nn
import numpy as np
import shap
from enformer_pytorch import str_to_one_hot
import json

import src.reglm.regression

from .base_optimizer import BaseOptimizer
import src.reglm.dataset, src.reglm.lightning, src.reglm.utils, src.reglm.metrics
import scripts.utils, scripts.motifs

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

class reinforce_shap_tfbs_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _init(self, cfg):
        self.task = cfg.task

        self.tfbs_lambda = cfg.tfbs_lambda

        motifs, bg = scripts.motifs.read_meme(
            cfg.meme_path
        )
        print(f"Total motifs: {len(motifs)}")
        sel = scripts.utils.load_csv(
            cfg.ppms_path
        ).Matrix_id.tolist()
        self.motif2idx = {name: i for i, name in enumerate(sel)}
        self.idx2motif = {i: name for i, name in enumerate(sel)}

        motifs = [m for m in motifs if m.name.decode() in sel]
        print(f"Selected motifs: {len(motifs)}")
        self.motifs = motifs
        self.bg = bg
        
        self.oracle = src.reglm.regression.EnformerModel.load_from_checkpoint(
            cfg.oracle_path
        )
        self.oracle.eval()

        self.agent = src.reglm.lightning.LightningModel.load_from_checkpoint(
            cfg.model_name_or_path
        )

        self.vocab = list(self.agent.label_stoi.keys()) + list(self.agent.base_stoi.keys())

        self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        self.oracle.to(self.device)
        self.agent.to(self.device)
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg.lr)

        self.predict = self.predict_enformer

        # read json from tfbs_reward_path
        self.tfbs_reward_dict = json.load(open(cfg.tfbs_reward_path, 'r'))

    def update(self, obs, rewards, nonterms, episode_lens, cfg, metrics, log):
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms)

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        
        loss = loss_pg 
        loss = loss_pg + self.alpha * logprobs.sum(0, keepdim=True).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        if log:
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['grad_norm'] = grad_norm.item() 
            metrics['smiles_len'] = episode_lens.float().mean().item()
            # metrics['loss_p'] = loss_p.item()
            metrics['alpha'] = self.alpha
            metrics['alpha_loss'] = alpha_loss.detach().item()
            print('logging!')
            wandb.log(metrics)

    def optimize(self, cfg):
        if cfg.wandb_log:
            wandb.init(
                # entity=cfg.wandb_entity,
                project="Hyena_RL",
                # dir=cfg.wandb_dir,
                name=cfg.wandb_run_name,
            )
            self.define_wandb_metrics()
        #set device
        self.device = torch.device(cfg.device)

        self._init(cfg)

        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        print('Start training ... ')
        while eval_strings < cfg.max_strings:

            with torch.no_grad():
                # sample experience
                labels = ["44"] * cfg.batch_size
                obs, rewards, nonterms, episode_lens = self.agent.get_data(labels, cfg.max_len)

            dna_list = []
            for dna in obs.cpu().numpy().T:
                dna_seq = self.agent.decode(dna, ignore_num=3)[0]
                dna_list.append(dna_seq)
            
            # ********* TFBS **********
            tfbs_sites = scripts.motifs.scan(dna_list, self.motifs, self.bg)
            tfbs_sites["start"] -= 1 # 1-based to 0-based
            tfbs_struc = scripts.motifs.generate_tfbs_structure(tfbs_sites)

            dna_one_hot = str_to_one_hot(dna_list).to(self.device)
            # importance = self.explainer.shap_values(dna_one_hot, check_additivity=False) # (bs, seq_len, 4)
            # importance = torch.tensor(importance, dtype=torch.float32, device=self.device)
            # importance *= dna_one_hot
            # importance = torch.sum(importance, dim=-1) # (bs, seq_len)
            # Adjust rewards based on TFBS contributions
            for idx, tfbs_info in enumerate(tfbs_struc):
                seq_id = list(tfbs_info.keys())[0]
                tfbs_list = tfbs_info[seq_id]
                
                # For each TFBS region
                for tfbs in tfbs_list:
                    start, end = tfbs['start'], tfbs['end']
                    
                    # Get the SHAP importance values for this TFBS region
                    # region_importance = importance[idx, start:end]
                    region_reward = self.tfbs_reward_dict[tfbs['Matrix_id']] * self.tfbs_lambda
                    # If all importance values in the region are negative, subtract from rewards

                    # rewards[start+2:end+2, idx] += region_reward
                    rewards[end+2-1, idx] += region_reward # end+2-1 because of 1-based indexing
                    
            score = np.array(self.predict(dna_list))
            scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.finish:
                print('max oracle hit')
                wandb.finish()
                sys.exit(0)

            train_steps += 1
            eval_strings += cfg.batch_size

            log = False
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                log = True
                metrics = dict()
                metrics['eval_strings'] = eval_strings
                metrics['mean_score'] = np.mean(score)
                metrics['max_score'] = np.max(score)
                metrics['min_score'] = np.min(score)
                metrics['mean_episode_lens'] = np.mean(episode_lens.tolist())
                metrics['max_episode_lens'] = np.max(episode_lens.tolist())
                metrics['min_episode_lens'] = np.min(episode_lens.tolist())
                wandb.log(metrics)

            # import pdb; pdb.set_trace()
            # rewards = rewards * scores
            rewards[-1, :] += (scores - 1).squeeze(0)
            self.update(obs, rewards, nonterms, episode_lens, cfg, metrics, log)

        print('max training string hit')
        wandb.finish()
        sys.exit(0)

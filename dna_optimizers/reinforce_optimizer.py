import os
import sys
import wandb
import hydra
import torch
from torch import optim
import numpy as np
import pandas as pd

import reglm_src.reglm.regression

from .base_optimizer import BaseOptimizer, evaluate
import reglm_src.reglm.dataset, reglm_src.reglm.lightning, reglm_src.reglm.utils, reglm_src.reglm.metrics

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

class reinforce_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._init(cfg)

    def _init(self, cfg):
        self.prefix_label =  cfg.prefix_label
        
        # agent
        # import pdb; pdb.set_trace()
        self.agent = src.reglm.lightning.LightningModel.load_from_checkpoint(
            cfg.model_name_or_path
        )
        self.agent.to(self.device)
        
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg.lr)

        self.vocab = list(self.agent.label_stoi.keys()) + list(self.agent.base_stoi.keys())

        self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        self.predict = self.predict_enformer
    
    def update(self, obs, rewards, nonterms, episode_lens, cfg, metrics, log):
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms)

        # print(logprobs)
        # print(act_probs)
        # print(logprobs.shape)
        # print(act_probs.shape)
        # exit()

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        
        
        #loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()
        # #loss = loss_pg #+ cfg.lp_coef * loss_p 
        # loss_p = self.alpha * logprobs.sum(0, keepdim=True).mean()
        loss_p = self.alpha * ( - (1 / logprobs.sum(0, keepdim=True)).mean())
        loss = loss_pg + loss_p

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # if log:
        #     metrics['pg_loss'] = loss_pg.item()       
        #     metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
        #     metrics['grad_norm'] = grad_norm.item() 
        #     metrics['smiles_len'] = episode_lens.float().mean().item()
        #     # metrics['loss_p'] = loss_p.item()
        #     metrics['alpha'] = self.alpha
        #     metrics['alpha_loss'] = alpha_loss.detach().item()
        #     print('logging!')
        #     wandb.log(metrics)

    def optimize(self, cfg, starting_sequences):

        # results' dataframe
        columns = ['round', 'sequence', 'true_score']
        df = pd.DataFrame(columns=columns)
        summary_columns = ["round", "top", "fitness", "diversity", "novelty"]
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
        
        # for i in range(cfg.epoch):
        #     self.update(init_tokens, init_rewards, init_nonterms, init_lens, cfg, dict(), False)
        
        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        print('Start training ... ')
        # while eval_strings < cfg.max_strings:
        for it in range(1, cfg.max_iter + 1):

            with torch.no_grad():
                # sample experience
                # labels = ["44"] * cfg.batch_size
                labels = [self.prefix_label] * cfg.batch_size
                obs, rewards, nonterms, episode_lens = self.agent.get_data(labels, cfg.max_len)

            dna_list = []
            for dna in obs.cpu().numpy().T:
                dna_seq = self.agent.decode(dna, ignore_num=len(self.prefix_label)+1)[0]
                assert len(dna_seq) == cfg.max_len
                dna_list.append(dna_seq)
                
            score = np.array(self.predict(dna_list))
            scores = torch.tensor(score, dtype=torch.float32, device=self.device)
             
            # if self.finish:
            #     print('max oracle hit')
            #     wandb.finish()
            #     sys.exit(0)

            train_steps += 1
            # eval_strings += cfg.batch_size

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

            round_df = pd.DataFrame({'round': [it]*len(dna_list), 'sequence': dna_list, 'true_score': score})
            df = pd.concat([df, round_df], ignore_index=True)
            round_results = evaluate(round_df, starting_sequences)
            round_results['round'] = it
            
            round_results_df = pd.DataFrame([round_results])
            summary_df = pd.concat([summary_df, round_results_df], ignore_index=True)
            
            if cfg.wandb_log:
                wandb.log(round_results)
            print(f"Round {it} finished")
            print(round_results)
            
        # save results
        # ./results/{cfg.task}_{cfg.level}_{cfg.seed}.csv
        os.makedirs('results', exist_ok=True)
        save_name = f'results/{cfg.task}_{cfg.level}_{cfg.seed}_{cfg.epoch}.csv'
        summary_name = f'results/{cfg.task}_{cfg.level}_{cfg.seed}_{cfg.epoch}_summary.csv'
        df.to_csv(save_name, index=False)
        summary_df.to_csv(summary_name, index=False)
        
        print('max training string hit')
        wandb.finish()
        sys.exit(0)
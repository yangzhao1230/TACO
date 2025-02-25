import numpy as np
import torch

class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, max_size, priority=True):
        self.memory = []
        self.max_size = max_size
        self.priority = priority

    def add_experience(self, dnas, obs, scores, rewards, nonterms, episode_lens):
        if self.max_size <= 0:
            return
        
        obs = obs.T # (batch, seq_len)
        nonterms = nonterms.T # (batch, seq_len)
        rewards = rewards.T # (batch, seq_len)
        episode_lens = episode_lens

        experience = zip(dnas, obs, scores, rewards, nonterms, episode_lens)
        self.memory.extend(experience)

        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, dna_list = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in dna_list:
                    idxs.append(i)
                    # smiles.append(exp[0])
                    dna_list.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]


            self.memory.sort(key=lambda x: x[2], reverse=True)
            self.memory = self.memory[:self.max_size]
            
    def sample(self, n, device):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[2]+1e-10 for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]

            obs = [x[1] for x in sample]
            scores = [x[2] for x in sample]
            rewards = [x[3] for x in sample]
            nonterms = [x[4] for x in sample]
            lens = [x[5] for x in sample]

        obs = torch.stack(obs).transpose(0,1) # (seq_len, batch)
        nonterms = torch.stack(nonterms).transpose(0,1) # (seq_len, batch)
        scores = torch.tensor(scores, dtype=torch.float32, device=device) # (batch)
        rewards = torch.stack(rewards).transpose(0,1) # (seq_len, batch)
        lens = torch.stack(lens)   
 
        return obs, scores, rewards, nonterms, lens

    def __len__(self):
        return len(self.memory)
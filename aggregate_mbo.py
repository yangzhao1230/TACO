import numpy as np
import pandas as pd
import itertools 
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cell', type=str, default='k562')
parser.add_argument('--level', type=str, default='easy')
parser.add_argument('--tfbs_lambda', type=float, default=0.0)

args = parser.parse_args()


cell = args.cell 
level = args.level
tfbs_lambda = args.tfbs_lambda

def distance(s1, s2):
    return sum([1 if i != j else 0 for i, j in zip(list(s1), list(s2))])

def get_fitness_info(cell):
    if cell == 'complex':
        length = 80
        min_fitness = 0
        max_fitness = 17
    elif cell == 'defined':
        length = 80
        min_fitness = 9
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

inits = pd.read_csv(f'data/{cell}/{level}.csv')
inits = inits.sort_values(by='target').iloc[:128]['sequence'].tolist()

length, min_fitness, max_fitness = get_fitness_info(cell) 

summary = []

if os.path.exists(f'summary/{cell}_{level}_{tfbs_lambda}_total.csv'):
    import sys
    sys.exit()
    
target = "oracle_score"
    
for run in tqdm(range(5)):
    ddir = f'results_mbo/{cell}_{level}_{tfbs_lambda}_{run}.csv'
    sequences = pd.read_csv(ddir)
    for r in tqdm(range(100, 101)):
        data = sequences[sequences['round']==r]
        data = data.sort_values(by=target,ascending=False).iloc[:128]
        top_fitness = data.iloc[:16][target].mean().item()
        median_fitness = data[target].median().item()
        seqs = data['sequence'].tolist()
        
        distances = []
        for s1, s2 in itertools.combinations(seqs, 2):
            distances.append(distance(s1, s2))
        diversity = np.median(distances)
        
        distances = []
        for j in seqs:
            dist_j = []
            for i in inits:
                dist_j.append(distance(i,j))
            distances.append(min(dist_j))
        novelty = np.median(distances)
        
        instance = [run, r, top_fitness, median_fitness, diversity, novelty]
        summary.append(instance)
        
results = pd.DataFrame(summary, columns=['run','round','top fitness', 'median fitness','diversity', 'novelty'])

import os
os.makedirs(f'summary/', exist_ok=True)
# results.to_csv(f'summary/{cell}_{level}_total.csv', index=False)
results.to_csv(f'summary/{cell}_{level}_{tfbs_lambda}_total.csv', index=False)
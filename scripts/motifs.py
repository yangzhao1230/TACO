import itertools
import os
import sys
import anndata
import numpy as np
import pandas as pd
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO
from collections import defaultdict
from enformer_pytorch.data import str_to_one_hot


def read_meme(meme_file):
    motifs = []
    motiffile = MotifFile(meme_file)
    
    while True:
        motif = motiffile.read()
        if motif is None:
            break
        motifs.append(motif)
        
    print(f"Read {len(motifs)} motifs")
    return motifs, motiffile.background


def scan(seq_df, motifs, bg, threshold=.001):
    
    # if seq_df is a dataframe, convert it to a list of Sequence objects
    if isinstance(seq_df, pd.DataFrame):
        sequences = [
            Sequence(row.Sequence, name=row.Index.encode())
            for row in seq_df.itertuples()
        ]
    else:
        # otherwise, assume it is a list of sequences, name is the idx
        sequences = [
            Sequence(seq, name=str(idx).encode())
            for idx, seq in enumerate(seq_df)
        ]

    d = defaultdict(list)
    fimo = FIMO(both_strands=True, threshold=threshold)
    
    for motif in motifs:
        match = fimo.score_motif(motif, sequences, bg).matched_elements
        for m in match:
            d['Matrix_id'].append(motif.name.decode())
            d['SeqID'].append(m.source.accession.decode())
            d['strand'].append(m.strand)
            d['score'].append(m.score)
            d['pval'].append(m.pvalue)
            d['qval'].append(m.qvalue)
            if m.strand == '-':
                d['start'].append(m.stop)
                d['end'].append(m.start)
            else:
                d['start'].append(m.start)
                d['end'].append(m.stop)
    
    return pd.DataFrame(d).set_index('SeqID')


def calculate_motif_counts(sites_df):
    return anndata.AnnData(
        pd.pivot_table(
            sites_df, values="start", index="SeqID", columns="Matrix_id", aggfunc="count"
        ).fillna(0)
    )


def accuracy_by_motif(seqs, sites, model):
    
    # Get motif locations
    sites = sites[sites.index.isin(seqs.index)]
    sites["positions"] = sites.apply(
        lambda row: set(list(range(row.start, row.end))), axis=1
    )

    # Get positions in and out of motifs
    positions = pd.DataFrame(
        sites.groupby("SeqID").positions.apply(list).apply(lambda x: set.union(*x))
    )
    seqs = seqs.merge(positions, left_index=True, right_on="SeqID", how="left")
    seqs['positions'] = seqs['positions'].fillna("").apply(set)
    seqs["all_positions"] = [set(range(len(seq))) for seq in seqs.Sequence]
    seqs["negative_positions"] = seqs.apply(lambda row: row["all_positions"].difference(row["positions"]), axis=1)

    # Compute accuracy in and out of motif
    seqs["In Motif"] = seqs.apply(
        lambda row: row.acc[list(row.positions)].mean(), axis=1
    )
    seqs["Out of Motif"] = seqs.apply(lambda row: row.acc[list(row.negative_positions)].mean(), axis=1)
    return seqs


def compare_motif_positions(sites, motifs):
    sites.index = sites.index.astype(str)
    sites.start = sites.start.astype(int)
    sites = sites[sites.Matrix_id.isin(motifs)]
    sites = sites.merge(
        all[["label", "Group", "SeqID"]], left_index=True, right_on="SeqID"
    )
    for m in motifs:
        print(m)
        kruskal_dunn(
            sites.loc[sites.Matrix_id == m, ["Matrix_id", "start", "Group"]], value_col="start", group_col="Group"
        )

def generate_tfbs_structure(tfbs_sites):
    
    result = []

    for seq_id, group in tfbs_sites.groupby("SeqID"):
        group_sorted = group.sort_values(["start", "end"])
        tfbs_list = []
        for _, row in group_sorted.iterrows():
            tfbs_list.append({"Matrix_id": row["Matrix_id"], "start": row["start"], "end": row["end"]})

        result.append({seq_id: tfbs_list})

    result = sorted(result, key=lambda x: list(x.keys())[0])

    return result
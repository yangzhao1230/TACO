import numpy as np
from polygraph.sequence import min_edit_distance


def filter_length(df, seq_col="seq", target=80):
    df['len'] = df[seq_col].apply(len)
    if isinstance(target, tuple):
        return df[(df['len'] >= target[0]) & (df['len'] <= target[1])]
    else:
        return df[df['len'] == target]


def drop_Ns(df, seq_col="seq"):
    return df[~df[seq_col].apply(lambda x: "N" in x)]
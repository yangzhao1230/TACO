import sys
import os
import numpy as np
import pandas as pd
import torch
from enformer_pytorch.data import str_to_one_hot
from ledidi import Ledidi
from torch import nn


INDEX_TO_BASE_HASH = {i: base for i, base in enumerate(["A", "C", "G", "T"])}


class DesignLoss(nn.Module):
    def __init__(self, specific):
        super().__init__()
        self.specific = specific

    def forward(self, preds, targets=None):
        if self.specific is None:
            return -preds.mean(axis=1).mean()
        else:
            non_specific = [x for x in range(preds.shape[1]) if x != self.specific]
            return (
                preds[:, non_specific].mean(axis=1) - preds[:, self.specific]
            ).mean()


class TransposeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.swapaxes(1, 2))


# Ledidi
def ledidi(
    start_seq,
    model,
    specific=None,
    max_iter=20000,
    device=0,
    num_workers=1,
    **kwargs,
):
    X = str_to_one_hot(start_seq).T.to(torch.device(device))  # 4, L
    model = TransposeModel(model).to(torch.device(device))
    designer = Ledidi(
        model,
        X.shape,
        output_loss=DesignLoss(specific),
        max_iter=max_iter,
        target=None,
        **kwargs,
    ).to(torch.device(device))

    X_hat = designer.fit_transform(X.unsqueeze(0), None).cpu()
    values, indices = X_hat.max(axis=1)
    return ["".join([INDEX_TO_BASE_HASH[i.tolist()] for i in idx]) for idx in indices]


def match(new, gen, pred_cols, group_col="Source"):
    cp = new.copy()
    matched = pd.DataFrame()

    for i in range(len(gen)):
        # Find closest match among new sequences
        mae = np.sum([np.abs(cp[col] - gen[col].iloc[i]) for col in pred_cols], 0)
        match_idx = np.argmin(mae)

        # Store
        matched = pd.concat([matched, cp.iloc[[match_idx],]])

        # Drop the start sequence
        if 'start_seq' in cp.columns:
            start_seq = cp.start_seq.iloc[match_idx]
            cp = cp[cp.start_seq != start_seq]

    return matched[["Sequence", group_col] + pred_cols]

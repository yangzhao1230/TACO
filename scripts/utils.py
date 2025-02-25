import pandas as pd
import torch


def load_csv(file, str_cols=["label"], **kwargs):
    return pd.read_csv(file, index_col=0, dtype={x: "str" for x in str_cols}, **kwargs)


def get_attn_scores(model, x):
    for l in model.trunk[:4]:
        x = l(x)
    
    x = model.trunk[4][0][0].fn[0](x)
    q = model.trunk[4][0][0].fn[1].to_q(x)
    k = model.trunk[4][0][0].fn[1].to_k(x)
    
    a = torch.stack([
        torch.nn.Softmax(1)((torch.matmul(qi, ki.T))/np.sqrt(10)) for qi, ki in zip(q, k)
    ])
    return a.detach().numpy()

import torch
import numpy as np
import scipy.sparse as sp
from baselines.utils import calculate_normalized_laplacian


def norm_adj(adj, k=0.8):
    assert isinstance(adj, np.ndarray)
    adj[adj < k] = 0
    L = calculate_normalized_laplacian(adj)
    L = sp.eye(adj.shape[0]) - L
    L = L.todense()
    L = torch.from_numpy(L)
    return L

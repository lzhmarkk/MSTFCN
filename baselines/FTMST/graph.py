import torch
import torch.nn as nn
import torch.nn.functional as fn
from baselines.MTGNN.MTGNN import graph_constructor


class GraphConstructor(nn.Module):
    def __init__(self, n_dim, n_nodes, top_k, dropout, spatial_func):
        super().__init__()

        self.n_dim = n_dim
        self.n_nodes = n_nodes
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.spatial_func = spatial_func

        if spatial_func == 'W':
            pass
        elif spatial_func == 'GCN':
            self.gc = graph_constructor(n_nodes, top_k, n_dim, 'cpu')
            self.idx = torch.arange(n_nodes)

    def forward(self):
        if self.spatial_func == 'W':
            g = torch.matmul(self.s_emb.weight, self.s_emb.weight.transpose(1, 0))
            g = torch.exp(3 * g)
        elif self.spatial_func == 'GCN':
            device = next(self.parameters()).device
            if self.gc.device != device:
                self.gc = self.gc.to(device)
                self.gc.device = device
                self.idx = self.idx.to(device)

            g = self.gc(self.idx)
        else:
            raise ValueError()

        return g

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
            self.w1 = nn.Parameter(torch.randn(n_nodes, n_nodes))
            self.w2 = nn.Parameter(torch.randn(n_nodes, n_nodes))
        elif spatial_func == 'EE':
            self.emb1 = nn.Embedding(n_nodes, n_dim)
            self.emb2 = nn.Embedding(n_nodes, n_dim)
        elif spatial_func == 'E1E2':
            self.emb1 = nn.Embedding(n_nodes, n_dim)
            self.emb2 = nn.Embedding(n_nodes, n_dim)
            self.emb3 = nn.Embedding(n_nodes, n_dim)
            self.emb4 = nn.Embedding(n_nodes, n_dim)
        elif spatial_func == 'E1E2-':
            self.emb1 = nn.Embedding(n_nodes, n_dim)
            self.emb2 = nn.Embedding(n_nodes, n_dim)
            self.emb3 = nn.Embedding(n_nodes, n_dim)
            self.emb4 = nn.Embedding(n_nodes, n_dim)
        elif spatial_func == 'M1M2':
            self.emb = nn.Embedding(n_nodes, n_dim)
            self.mlp1 = nn.Linear(n_dim, n_dim)
            self.mlp2 = nn.Linear(n_dim, n_dim)
            self.mlp3 = nn.Linear(n_dim, n_dim)
            self.mlp4 = nn.Linear(n_dim, n_dim)
        elif spatial_func == 'M1M2-':
            self.emb = nn.Embedding(n_nodes, n_dim)
            self.mlp1 = nn.Linear(n_dim, n_dim)
            self.mlp2 = nn.Linear(n_dim, n_dim)
            self.mlp3 = nn.Linear(n_dim, n_dim)
            self.mlp4 = nn.Linear(n_dim, n_dim)
        elif spatial_func == 'GCN':
            self.gc_real = graph_constructor(n_nodes, top_k, n_dim, 'cpu')
            self.gc_imag = graph_constructor(n_nodes, top_k, n_dim, 'cpu')
            self.idx = torch.arange(n_nodes)

    def forward(self):
        if self.spatial_func == 'W':
            g1 = self.w1
            g2 = self.w2
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'EE':
            g1 = torch.mm(self.emb1.weight, self.emb1.weight.transpose(1, 0))
            g2 = torch.mm(self.emb2.weight, self.emb2.weight.transpose(1, 0))
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'E1E2':
            g1 = torch.mm(self.emb1.weight, self.emb2.weight.transpose(1, 0))
            g2 = torch.mm(self.emb3.weight, self.emb4.weight.transpose(1, 0))
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'E1E2-':
            g1 = torch.mm(self.emb1.weight, self.emb2.weight.transpose(1, 0)) - torch.mm(self.emb2.weight, self.emb1.weight.transpose(1, 0))
            g2 = torch.mm(self.emb3.weight, self.emb4.weight.transpose(1, 0)) - torch.mm(self.emb4.weight, self.emb3.weight.transpose(1, 0))
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'M1M2':
            m1 = self.mlp1(self.emb.weight)
            m2 = self.mlp2(self.emb.weight)
            m3 = self.mlp3(self.emb.weight)
            m4 = self.mlp4(self.emb.weight)
            g1 = torch.mm(m1, m2.transpose(1, 0))
            g2 = torch.mm(m3, m4.transpose(1, 0))
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'M1M2-':
            m1 = self.mlp1(self.emb.weight)
            m2 = self.mlp2(self.emb.weight)
            m3 = self.mlp3(self.emb.weight)
            m4 = self.mlp4(self.emb.weight)
            g1 = torch.mm(m1, m2.transpose(1, 0)) - torch.mm(m2, m1.transpose(1, 0))
            g2 = torch.mm(m3, m4.transpose(1, 0)) - torch.mm(m4, m3.transpose(1, 0))
            g = [fn.relu(fn.tanh(g1)), fn.relu(fn.tanh(g2))]
        elif self.spatial_func == 'GCN':
            device = next(self.parameters()).device
            if self.gc_real.device != device:
                self.gc_real = self.gc_real.to(device)
                self.gc_real.device = device
                self.idx = self.idx.to(device)
                self.gc_imag = self.gc_imag.to(device)
                self.gc_imag.device = device

            g = [self.gc_real(self.idx), self.gc_imag(self.idx)]
        else:
            raise ValueError()

        return g

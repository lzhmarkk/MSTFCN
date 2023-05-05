import torch
import torch.nn as nn
import torch.nn.functional as fn
from baselines.MTGNN.MTGNN import dilated_inception


class FTLayer(nn.Module):
    def __init__(self, n_dim, n_nodes, start_length, end_length, dropout):
        super().__init__()

        self.n_dim = n_dim
        self.n_nodes = n_nodes
        self.start_length = start_length
        self.end_length = end_length
        self.dropout = nn.Dropout(dropout)

        # temporal
        self.complex_weight = nn.Parameter(torch.randn(1, n_dim, 1, start_length // 2 + 1, 2) * 0.02, requires_grad=True)
        self.complex_bias = nn.Parameter(torch.randn(1, n_dim, 1, start_length // 2 + 1, 2) * 0.01, requires_grad=True)
        self.t_fc = nn.Linear(start_length, end_length, bias=True)
        self.t_mlp = nn.Sequential(nn.Linear(start_length, end_length),
                                   nn.GELU(),
                                   nn.Linear(end_length, end_length))
        self.filter_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)
        self.gate_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)

        # spatial
        pass

        self.s_emb = nn.Embedding(n_nodes, n_dim)
        nn.init.xavier_uniform_(self.s_emb.weight)

        self.ln = nn.LayerNorm([n_dim, n_nodes, end_length], elementwise_affine=True)

    def fft(self, x):
        x = torch.fft.rfft(x, dim=3, norm='ortho')
        return x

    def ifft(self, x):
        x = torch.fft.irfft(x, n=self.end_length, dim=3, norm='ortho')
        return x

    def norm(self, x):
        return self.ln(x)

    def forward(self, x):
        # filter = self.filter_convs(x)
        # filter = torch.tanh(filter)
        # gate = self.gate_convs(x)
        # gate = torch.sigmoid(gate)
        # x = filter * gate
        x = self.t_mlp(x)

        # (B, C, N, T)
        # temporal
        # x = self.fft(x)
        # weight = torch.view_as_complex(self.complex_weight)
        # bias = torch.view_as_complex(self.complex_bias)
        # x = x * weight + bias

        # spatial
        # g = torch.matmul(self.s_emb.weight, self.s_emb.weight.transpose(1, 0))
        # g = torch.exp(3 * g)
        # x = torch.einsum('bcnt, nm->bcmt', x, g)

        # x = self.ifft(x)
        return x

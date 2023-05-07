import torch
import torch.nn as nn
import torch.nn.functional as fn
from baselines.MTGNN.MTGNN import dilated_inception


class FTLayer(nn.Module):
    def __init__(self, n_dim, n_nodes, start_length, end_length, temporal_func, spatial_func, dropout):
        super().__init__()

        self.n_dim = n_dim
        self.n_nodes = n_nodes
        self.start_length = start_length
        self.end_length = end_length
        self.temporal_func = temporal_func
        self.spatial_func = spatial_func
        self.dropout = nn.Dropout(dropout)

        # temporal
        if temporal_func == 'FC':
            self.t_fc = nn.Linear(start_length, end_length)
        elif temporal_func == 'MLP':
            self.t_mlp = nn.Sequential(nn.Linear(start_length, end_length),
                                       nn.GELU(),
                                       nn.Linear(end_length, end_length))
        elif temporal_func == 'TCN':
            self.filter_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)
            self.gate_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)
        elif temporal_func == 'FFT':
            self.n_fft = 1  # relu between mlp if 2
            self.weight = nn.Parameter(torch.randn((n_dim, n_dim, start_length // 2 + 1), dtype=torch.cfloat))
            self.bias = nn.Parameter(torch.randn((1, n_dim, 1, start_length // 2 + 1), dtype=torch.cfloat))
            if self.n_fft == 2:
                self.weight2 = nn.Parameter(torch.randn((n_dim, n_dim, start_length // 2 + 1), dtype=torch.cfloat))
                self.bias2 = nn.Parameter(torch.randn((1, n_dim, 1, start_length // 2 + 1), dtype=torch.cfloat))
            # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # nn.init.uniform_(self.bias, -bound, bound)

        # spatial
        if spatial_func == 'Mul':
            self.s_emb = nn.Embedding(n_nodes, n_dim)
            nn.init.xavier_uniform_(self.s_emb.weight)
        elif spatial_func == 'GCN':
            from baselines.MTGNN.MTGNN import mixprop
            self.gconv1 = mixprop(n_dim, n_dim, 3, dropout, 0.05)
            self.gconv2 = mixprop(n_dim, n_dim, 3, dropout, 0.05)

        self.ln = nn.LayerNorm([n_dim, n_nodes, end_length], elementwise_affine=True)

    def norm(self, x):
        return self.ln(x)

    def forward(self, x, g):
        # (B, C, N, T), (N, N)
        # temporal
        if self.temporal_func == 'FC':
            x = torch.tanh(self.t_fc(x))
        elif self.temporal_func == 'MLP':
            x = torch.tanh(self.t_mlp(x))
        elif self.temporal_func == 'TCN':
            filter = torch.tanh(self.filter_convs(x))
            gate = torch.sigmoid(self.gate_convs(x))
            x = filter * gate
        elif self.temporal_func == 'FFT':
            x = torch.fft.rfft(x, dim=3, norm='ortho')
            x = torch.einsum('bint,iot->bont', x, self.weight)
            x = x + self.bias
            if self.n_fft == 2:
                x = torch.relu(x.real) + 1.j * torch.relu(x.imag)
                x = torch.einsum('bint,iot->bont', x, self.weight2)
                x = x + self.bias2
        else:
            pass

        if self.spatial_func == 'Mul':
            if torch.is_complex(x):
                real = torch.einsum('bcnt, nm->bcmt', x.real, g)
                imag = torch.einsum('bcnt, nm->bcmt', x.imag, g)
                x = real + 1.j * imag
            else:
                x = torch.einsum('bcnt, nm->bcmt', x, g)

        elif self.spatial_func == 'GCN':
            if torch.is_complex(x):
                real = self.gconv1(x.real, g) + self.gconv2(x.real, g.transpose(1, 0))
                imag = self.gconv1(x.imag, g) + self.gconv2(x.imag, g.transpose(1, 0))
                x = real + 1.j * imag
            else:
                x = self.gconv1(x, g) + self.gconv2(x, g.transpose(1, 0))

        else:
            pass

        if self.temporal_func == 'FFT':
            x = torch.fft.irfft(x, n=self.end_length, dim=3, norm='ortho')
            x = torch.tanh(x)
        return x

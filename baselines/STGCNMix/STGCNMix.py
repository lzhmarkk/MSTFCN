import torch
import torch.nn as nn
from ..STGCN.STGCN import STGCN
from ..CRGNN.TimeEncoder import TimeEncoder


class STGCNMix(nn.Module):
    def __init__(self, device, adj_mx, input_dim, output_dim, horizon, window, num_nodes, spatial_channels,
                 hidden_channel, add_time):
        super(STGCNMix, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.n_nodes = num_nodes
        self.device = device
        self.adj_mx = torch.from_numpy(adj_mx).float().to(device)
        self.add_time = add_time

        _dim = 0
        self.in_split, self.out_split = [], []
        for d in self.input_dim:
            self.in_split.append((_dim, _dim + d))
            _dim += d
        _dim = 0
        for d in self.output_dim:
            self.out_split.append((_dim, _dim + d))
            _dim += d

        models = []
        for i in range(self.n_mix):
            models.append(STGCN(adj_mx=adj_mx, device=device, num_nodes=num_nodes, input_dim=input_dim[i],
                                output_dim=output_dim[i], window=window, horizon=horizon,
                                spatial_channels=spatial_channels, hidden_channel=hidden_channel))
        self.models = nn.ModuleList(models)

        if add_time:
            self.t_enc = TimeEncoder(hidden_channel, self.window)

        self.weight = nn.Parameter(torch.randn([self.n_mix, self.n_mix, num_nodes]), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, **kwargs):
        bs = input.shape[0]
        if self.add_time:
            input, time = input[..., :-2], input[..., -2:]

        input = input.transpose(2, 1)
        x = [input[..., p[0]: p[1]] for p in self.in_split]  # (B, N, T, C) * n_mix

        for i in range(self.n_mix):
            x[i] = self.models[i].block1(x[i], self.adj_mx)
            x[i] = self.models[i].block2(x[i], self.adj_mx)

        # merge mix
        for i in range(self.n_mix):
            h = []
            for j in range(self.n_mix):
                w = self.weight[i, j].reshape(1, self.n_nodes, 1, 1)
                if i == j:
                    w = w + 1

                h.append(w * x[j])
            h = torch.stack(h, 0).sum(0)
            x[i] = h

        for i in range(self.n_mix):
            x[i] = self.models[i].last_temporal(x[i])
            x[i] = self.models[i].fully(x[i].reshape(bs, self.n_nodes, -1))  # (B, N, T * C)
            x[i] = x[i].reshape(bs, self.n_nodes, self.horizon, self.output_dim[i]).permute(0, 2, 1, 3)

        x = torch.cat(x, -1)
        return x

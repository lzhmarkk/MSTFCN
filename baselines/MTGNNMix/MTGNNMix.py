from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..MTGNN.MTGNN import MTGNN
from ..CRGNN.TimeEncoder import TimeEncoder


class MTGNNMix(nn.Module):
    def __init__(self, device, num_nodes, gcn_depth, dropout, input_dim, output_dim,
                 window, horizon, subgraph_size, node_dim, tanhalpha, propalpha, dilation_exponential,
                 layers, residual_channels, conv_channels, skip_channels, end_channels, add_time):
        super(MTGNNMix, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.n_nodes = num_nodes
        self.device = device
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
            models.append(MTGNN(device=device, adj_mx=torch.eye(num_nodes), gcn_true=True, buildA_true=True,
                                num_nodes=num_nodes, gcn_depth=gcn_depth, dropout=dropout,
                                input_dim=input_dim[i], output_dim=output_dim[i], window=window, horizon=horizon,
                                subgraph_size=subgraph_size, node_dim=node_dim, tanhalpha=tanhalpha,
                                propalpha=propalpha,
                                dilation_exponential=dilation_exponential, layers=layers,
                                residual_channels=residual_channels, conv_channels=conv_channels,
                                skip_channels=skip_channels,
                                end_channels=end_channels, add_time=False))
        self.models = nn.ModuleList(models)
        self.idx = torch.arange(self.n_nodes).to(self.device)

        if add_time:
            self.t_enc = TimeEncoder(skip_channels, self.window)

        self.weight = nn.Parameter(torch.randn([self.n_mix, self.n_mix, num_nodes]), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, **kwargs):
        bs = input.shape[0]
        if self.add_time:
            input, time = input[..., :-2], input[..., -2:]

        input = input.transpose(3, 1)
        if self.models[0].seq_length < self.models[0].receptive_field:
            input = nn.functional.pad(input, (self.models[0].receptive_field - self.models[0].seq_length, 0, 0, 0))
        x = [input[:, p[0]: p[1]] for p in self.in_split]  # (n_mix, bs, in_dim, n_nodes, window)

        for i in range(self.n_mix):
            adp = self.models[i].gc(self.idx)

            skip = self.models[i].skip0(F.dropout(x[i], self.models[i].dropout, training=self.training))
            x[i] = self.models[i].start_conv(x[i])
            for l in range(self.models[i].layers):
                residual = x[i]
                filter = self.models[i].filter_convs[l](x[i])
                filter = torch.tanh(filter)
                gate = self.models[i].gate_convs[l](x[i])
                gate = torch.sigmoid(gate)
                x[i] = filter * gate
                x[i] = F.dropout(x[i], self.models[i].dropout, training=self.training)
                s = x[i]
                s = self.models[i].skip_convs[l](s)
                skip = s + skip
                x[i] = self.models[i].gconv1[l](x[i], adp) + self.models[i].gconv2[l](x[i], adp.transpose(1, 0))

                x[i] = x[i] + residual[:, :, :, -x[i].size(3):]
                x[i] = self.models[i].norm[l](x[i], self.idx)

            skip = self.models[i].skipE(x[i]) + skip
            x[i] = F.relu(skip)  # (B, C, N, T)

        # merge mix
        for i in range(self.n_mix):
            h = []
            for j in range(self.n_mix):
                w = self.weight[i, j].reshape(1, 1, self.n_nodes, 1)
                if i == j:
                    w = w + 1

                h.append(w * x[j])
            h = torch.stack(h, 0).sum(0)
            x[i] = h

        for i in range(self.n_mix):
            if self.add_time:
                x[i] = self.t_enc(x[i], time)
            x[i] = F.relu(self.models[i].end_conv_1(x[i]))
            x[i] = self.models[i].end_conv_2(x[i])

            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.n_nodes, 1).squeeze(-1).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

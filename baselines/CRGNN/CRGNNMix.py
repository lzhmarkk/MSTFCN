from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CRGNN import CRGNN
from .GraphConstructor import CrossRelationGraphConstructor
from .TimeEncoder import TimeEncoder


class CRGNNMix(nn.Module):
    def __init__(self, device, num_nodes, gcn_depth, dropout, input_dim, output_dim,
                 window, horizon, subgraph_size, node_dim, tanhalpha, propalpha, dilation_exponential,
                 layers, residual_channels, conv_channels, skip_channels, end_channels, cross, temporal_func):
        super(CRGNNMix, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.n_nodes = num_nodes
        self.device = device
        self.layers = layers
        self.cross = cross

        _dim = 0
        self.in_split, self.out_split = [], []
        for d in self.input_dim:
            self.in_split.append((_dim, _dim + d))
            _dim += d
        _dim = 0
        for d in self.output_dim:
            self.out_split.append((_dim, _dim + d))
            _dim += d

        self.graph_constructor = CrossRelationGraphConstructor(nnodes=num_nodes, k=subgraph_size, dim=node_dim,
                                                               device=device, alpha=tanhalpha, n_mix=self.n_mix,
                                                               cross_relation=self.cross)

        models = []
        for i in range(self.n_mix):
            models.append(CRGNN(device=device, num_nodes=num_nodes, gcn_depth=gcn_depth, dropout=dropout,
                                input_dim=input_dim[i], output_dim=output_dim[i], window=window, horizon=horizon,
                                propalpha=propalpha, dilation_exponential=dilation_exponential, layers=layers,
                                residual_channels=residual_channels, conv_channels=conv_channels,
                                skip_channels=skip_channels, end_channels=end_channels,
                                temporal_func=temporal_func))
        self.models = nn.ModuleList(models)

        self.time_encoder = TimeEncoder(dim=skip_channels, length=window)
        self.dropout = nn.Dropout(dropout)

        if self.cross:
            self.cross_weight = nn.Parameter(torch.randn(self.n_mix, self.n_mix), requires_grad=True)

    def pad_sequence(self, x):
        receptive_field = self.models[0].receptive_field
        if self.window < receptive_field:
            return F.pad(x, (receptive_field - self.window, 0, 0, 0))
        else:
            return x

    def forward(self, input, **kwargs):
        x, time = input[..., :-2], input[..., -2:]
        bs = x.shape[0]

        graphs = self.graph_constructor()  # (n_mix, n_mix, n_nodes, n_nodes)

        x = x.transpose(3, 1)
        x = self.pad_sequence(x)  # (bs, mix_in_dim, n_nodes, window)
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (n_mix, bs, in_dim, n_nodes, window)

        output = []
        for i in range(self.n_mix):
            _output = self.models[i].skip0(self.dropout(x[i]))
            output.append(_output)

        for i in range(self.n_mix):
            x[i] = self.models[i].start_conv(x[i])  # (bs, res_channel, n_nodes, recep_field)

        for l in range(self.layers):
            residuals = []
            for i in range(self.n_mix):
                residuals.append(x[i])

            for i in range(self.n_mix):
                x[i] = self.models[i].temporal_layer(x[i], l)

                # _out = self.models[i].skip_convs[l](x[i])  # (bs, skp_channel, n_nodes, 1)
                # output[i] = _out + output[i]

            for i in range(self.n_mix):
                h = []
                for j in range(self.n_mix):
                    g, w = graphs[i, j], self.cross_weight[i, j]
                    if i == j:
                        w = w + 1
                    # (bs, res_channel, n_nodes, recep_filed + (1 - max_ker_size) * i)
                    _h = self.models[i].spatial_layer(x[j], g, l)
                    _h = _h * w
                    h.append(_h)
                h = torch.stack(h, 0).sum(0)
                x[i] = h

            for i in range(self.n_mix):
                x[i] = self.models[i].channel_layer(x[i], l)

            for i in range(self.n_mix):
                # (bs, res_channel, n_nodes, recep_filed + (1 - max_ker_size) * i)
                # x[i] = x[i] + residuals[i][:, :, :, -x[i].size(3):]
                _out = self.models[i].skip_convs[l](x[i])  # (bs, skp_channel, n_nodes, 1)
                output[i] = _out + output[i]

        for i in range(self.n_mix):
            # output[i] = self.models[i].skipE(x[i]) + output[i]
            x[i] = F.relu(output[i])

            # time encoding
            x[i] = self.time_encoder(x[i], time)
            x[i] = self.models[i].end_conv(x[i])

            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.n_nodes, 1).squeeze(-1).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

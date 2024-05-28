import torch
import torch.nn as nn
import torch.nn.functional as F
from .MSTFCNMode import MSTFCNMode
from .GraphConstructor import CrossRelationGraphConstructor
from .TimeEncoder import TimeEncoder


class MSTFCN(nn.Module):
    def __init__(self, device, num_nodes, gcn_depth, dropout, input_dim, output_dim,
                 window, horizon, subgraph_size, node_dim, tanhalpha, propalpha,
                 layers, residual_channels, conv_channels, skip_channels, end_channels, cross, temporal_func, add_time):
        super(MSTFCN, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.n_nodes = num_nodes
        self.device = device
        self.layers = layers
        self.cross = cross
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

        self.graph_constructor = CrossRelationGraphConstructor(nnodes=num_nodes, k=subgraph_size, dim=node_dim,
                                                               device=device, alpha=tanhalpha, n_mix=self.n_mix,
                                                               cross_relation=self.cross)

        models = []
        for i in range(self.n_mix):
            models.append(MSTFCNMode(device=device, num_nodes=num_nodes, gcn_depth=gcn_depth, dropout=dropout,
                                     input_dim=input_dim[i], output_dim=output_dim[i], window=window, horizon=horizon,
                                     propalpha=propalpha, layers=layers,
                                     residual_channels=residual_channels, conv_channels=conv_channels,
                                     skip_channels=skip_channels, end_channels=end_channels,
                                     temporal_func=temporal_func, add_time=add_time))
        self.models = nn.ModuleList(models)

        if self.add_time:
            self.time_encoder = TimeEncoder(dim=residual_channels // 2, length=window)
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

        x = x.transpose(3, 1)  # (bs, mix_in_dim, n_nodes, window)
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (n_mix, bs, in_dim, n_nodes, window)
        b = [self.time_encoder.embed_time(time) for _ in range(self.n_mix)]  # (n_mix, bs, res_channel, n_nodes, window)

        output, output_b = [], []
        for i in range(self.n_mix):
            output.append(self.models[i].skip0(self.dropout(x[i])))
            output_b.append(0.)

        for i in range(self.n_mix):
            x[i] = self.models[i].start_conv(x[i])  # (bs, res_channel, n_nodes, recep_field)

        for l in range(self.layers):
            residuals = []
            for i in range(self.n_mix):
                residuals.append(x[i])

            # temporal
            for i in range(self.n_mix):
                x[i] = self.models[i].temporal_layer(x[i], l)
                b[i] = self.models[i].temporal_layer(b[i], l)

            # spatial & mode
            for i in range(self.n_mix):
                h = []
                for j in range(self.n_mix):
                    g, w = graphs[i, j], (self.cross_weight[i, j] if self.cross else 0)
                    if i == j:
                        w = w + 1
                    # (bs, res_channel, n_nodes, recep_filed + (1 - max_ker_size) * i)
                    _h = self.models[i].spatial_layer(x[j], g, l)
                    _h = _h * w
                    h.append(_h)
                h = torch.stack(h, 0).sum(0)
                x[i] = h

            # channel
            for i in range(self.n_mix):
                h = self.models[i].channel_layer(torch.cat([x[i], b[i]], dim=1), l)
                x[i], b[i] = h.chunk(2, dim=1)

            # to output
            for i in range(self.n_mix):
                # (bs, skp_channel, n_nodes, 1)
                output[i] += self.models[i].skip_convs[l](self.dropout(x[i]))
                output_b[i] += self.models[i].skip_convs[l](self.dropout(b[i]))

        for i in range(self.n_mix):
            x[i] = torch.cat([output[i], output_b[i]], dim=1)
            x[i] = F.relu(x[i])
            x[i] = self.models[i].end_conv(x[i])
            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.n_nodes, 1).squeeze(-1).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

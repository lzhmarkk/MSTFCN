from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CRGNN import CRGNN


class CrossRelationGraphConstructor(nn.Module):
    def __init__(self, n_mix, nnodes, k, dim, device, alpha=3, cross_relation=True, full_graph=False):
        super(CrossRelationGraphConstructor, self).__init__()
        self.n_mix = n_mix
        self.n_nodes = nnodes
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.cross = cross_relation
        self.full_graph = full_graph

        # use different mix emb
        self.emb1 = nn.ModuleList([nn.Embedding(self.n_nodes, self.dim)] * self.n_mix)
        # use mlp to identify mix emb
        self.emb2 = nn.Embedding(self.n_nodes, self.dim)
        self.emb2_mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=True)), nn.ReLU()] * self.n_mix)

        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)

    def forward(self):
        adjs = []
        for i, nodevec1 in enumerate(self.emb1):
            adjs_row = []
            # nodevec1 = nodevec1(self.emb2.weight)
            nodevec1 = nodevec1.weight
            for j, nodevec2 in enumerate(self.emb1):
                # nodevec2 = nodevec2(self.emb2.weight)
                nodevec2 = nodevec2.weight
                if not self.cross and i != j:
                    adj = torch.zeros(self.n_nodes, self.n_nodes).to(self.device)
                    adjs_row.append(adj)
                else:
                    nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
                    nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

                    a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
                    adj = torch.relu(torch.tanh(self.alpha * a))
                    if self.full_graph:
                        adjs_row.append(adj)
                    else:
                        mask = torch.zeros(self.n_nodes, self.n_nodes).to(self.device)
                        mask.fill_(float('0'))
                        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
                        mask.scatter_(1, t1, s1.fill_(1))
                        adj = adj * mask
                        adjs_row.append(adj)
            adjs.append(torch.stack(adjs_row, 0))  # (n_mix, n_nodes, n_nodes)
        return torch.stack(adjs, 0)  # (n_mix, n_mix, n_nodes, n_nodes)


class TimeEncoder(nn.Module):
    def __init__(self, dim, len):
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.len = len

        self.time_day_mlp = nn.Conv2d(dim, dim, kernel_size=(1, len))
        self.time_week_mlp = nn.Conv2d(dim, dim, kernel_size=(1, len))
        self.time_day_emb = nn.Parameter(torch.randn(48, dim))
        self.time_week_emb = nn.Parameter(torch.randn(7, dim))

    def forward(self, x, time):
        time_in_day_emb = self.time_day_emb[(time[..., -2] * 48).long()].permute(0, 3, 2, 1)
        day_in_week_emb = self.time_week_emb[(time[..., -1]).long()].permute(0, 3, 2, 1)
        time_in_day_emb = self.time_day_mlp(time_in_day_emb)
        day_in_week_emb = self.time_week_mlp(day_in_week_emb)
        x = torch.cat([x, time_in_day_emb, day_in_week_emb], 1)
        return x


class CRGNNMix(nn.Module):
    def __init__(self, device, adj_mx, gcn_true, buildA_true, num_nodes, gcn_depth, dropout, input_dim, output_dim,
                 window, horizon, subgraph_size, node_dim, tanhalpha, propalpha, dilation_exponential,
                 layers, residual_channels, conv_channels, skip_channels, end_channels, cross):
        super(CRGNNMix, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.dropout = dropout
        self.idx = torch.arange(num_nodes).to(device)
        self.skip_channels = skip_channels
        self.seq_length = window
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
                                                               device=device, alpha=tanhalpha, n_mix=self.n_mix)

        models = []
        for i in range(self.n_mix):
            models.append(CRGNN(device=device, adj_mx=adj_mx, gcn_true=gcn_true, buildA_true=buildA_true,
                          num_nodes=num_nodes, gcn_depth=gcn_depth, dropout=dropout,
                          input_dim=input_dim[i], output_dim=output_dim[i], window=window, horizon=horizon,
                          subgraph_size=subgraph_size, node_dim=node_dim, tanhalpha=tanhalpha,
                          propalpha=propalpha, dilation_exponential=dilation_exponential, layers=layers,
                          residual_channels=residual_channels, conv_channels=conv_channels, skip_channels=skip_channels,
                          end_channels=end_channels))
        self.models = nn.ModuleList(models)

        self.time_encoder = TimeEncoder(dim=self.skip_channels, len=self.seq_length)

        if self.cross:
            self.cross_weight = nn.Parameter(torch.randn(self.n_mix, self.n_mix), requires_grad=True)

    def pad_sequence(self, x):
        receptive_field = self.models[0].receptive_field
        if self.seq_length < receptive_field:
            return F.pad(x, (receptive_field - self.seq_length, 0, 0, 0))
        else:
            return x

    def forward(self, input, **kwargs):
        x, time = input[..., :-2], input[..., -2:]
        bs = x.shape[0]

        graphs = self.graph_constructor()  # (n_mix, n_mix, n_nodes, n_nodes)
        # graphs = []
        # for i in range(self.n_mix):
        #     _g = self.models[i].build_graph(x)
        #     graphs.append(_g)

        x = x.transpose(3, 1)
        x = self.pad_sequence(x)  # (bs, mix_in_dim, n_nodes, window)
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (n_mix, bs, in_dim, n_nodes, window)

        output = []
        for i in range(self.n_mix):
            _output = self.models[i].skip0(F.dropout(x[i], self.dropout, training=self.training))
            output.append(_output)

        for i in range(self.n_mix):
            x[i] = self.models[i].start_conv(x[i])  # (bs, res_channel, n_nodes, recep_field)

        for l in range(self.layers):
            residuals = []
            for i in range(self.n_mix):
                residuals.append(x[i])
                x[i] = self.models[i].temporal_conv(x[i], l)

                _out = self.models[i].skip_convs[l](x[i])  # (bs, skp_channel, n_nodes, 1)
                output[i] = _out + output[i]

            for i in range(self.n_mix):
                h = []
                for j in range(self.n_mix):
                    g, w = graphs[i, j], self.cross_weight[i, j]
                    if i == j:
                        w = w + 1
                    # (bs, res_channel, n_nodes, recep_filed + (1 - max_ker_size) * i)
                    _h = self.models[i].spatial_conv(x[j], g, l)
                    _h = _h * w
                    h.append(_h)
                h = torch.stack(h, 0).sum(0)
                x[i] = h
            # for i in range(self.n_mix):
            #     # todo add cross-relation graph here
            #     x[i] = self.models[i].spatial_conv(x[i], graphs[i], l)

            for i in range(self.n_mix):
                # (bs, res_channel, n_nodes, recep_filed + (1 - max_ker_size) * i)
                x[i] = x[i] + residuals[i][:, :, :, -x[i].size(3):]
                x[i] = self.models[i].norm[l](x[i], self.idx)

        for i in range(self.n_mix):
            output[i] = self.models[i].skipE(x[i]) + output[i]
            x[i] = F.relu(output[i])

            # time encoding
            x[i] = self.time_encoder(x[i], time)
            x[i] = self.models[i].end_conv(x[i])

            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.n_nodes, 1).squeeze(dim=-1).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

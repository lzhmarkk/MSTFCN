from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..GWNet.GWNet import GWNet, linear, nconv
from ..MSTFCN.TimeEncoder import TimeEncoder
from ..MSTFCN.GraphConstructor import CrossRelationGraphConstructor


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((order + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, a):
        out = [x]
        for _ in range(self.order):
            x = self.nconv(x, a)
            out.append(x)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNetMix(nn.Module):
    def __init__(self, device, num_nodes, dropout, input_dim, output_dim, window, horizon, subgraph_size,
                 node_dim, tanhalpha, layers, adj_mxs, adjtype, randomadj, aptonly, nhid, kernel_size,
                 blocks, add_time):
        super(GWNetMix, self).__init__()

        self.n_mix = len(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.window = window
        self.n_nodes = num_nodes
        self.adj_mxs = [torch.from_numpy(a).float().to(device) for a in adj_mxs]
        self.device = device
        self.blocks = blocks
        self.layers = layers
        self.add_time = add_time
        assert not add_time

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
            m = GWNet(adj_mx=adj_mxs[i], device=device, adjtype=adjtype, randomadj=randomadj,
                      aptonly=aptonly, nhid=nhid, input_dim=input_dim[i], output_dim=output_dim[i],
                      num_nodes=num_nodes, kernel_size=kernel_size, horizon=horizon, window=window,
                      dropout=dropout, blocks=blocks, layers=layers, gcn_bool=True, addaptadj=True)
            m.gconv = nn.ModuleList([gcn(nhid, nhid, dropout)] * 3)
            models.append(m)
        self.models = nn.ModuleList(models)
        self.idx = torch.arange(self.n_nodes).to(self.device)

        self.graph_constructor = CrossRelationGraphConstructor(nnodes=num_nodes, k=subgraph_size, dim=node_dim,
                                                               device=device, alpha=tanhalpha, n_mix=self.n_mix,
                                                               cross_relation=True)

        self.cross_weight = nn.Parameter(torch.randn([self.n_mix, self.n_mix]), requires_grad=True)
        # nn.init.xavier_uniform_(self.cross_weight)

    def forward(self, input, **kwargs):
        bs = input.shape[0]
        graphs = self.graph_constructor()  # (n_mix, n_mix, n_nodes, n_nodes)

        input = input.transpose(3, 1)
        if input.shape[3] < self.models[0].receptive_field:
            input = nn.functional.pad(input, (self.models[0].receptive_field - input.shape[3], 0, 0, 0))
        x = [input[:, p[0]: p[1]] for p in self.in_split]  # (n_mix, bs, in_dim, n_nodes, window)
        skip = [0.] * self.n_mix

        for i in range(self.n_mix):
            x[i] = self.models[i].start_conv(x[i])

        for l in range(self.blocks * self.layers):
            residual = [0.] * self.n_mix

            for i in range(self.n_mix):
                residual[i] = x[i]
                filter = self.models[i].filter_convs[l](x[i])
                filter = torch.tanh(filter)
                gate = self.models[i].gate_convs[l](x[i])
                gate = torch.sigmoid(gate)
                x[i] = filter * gate

                s = self.models[i].skip_convs[l](x[i])
                if not l == 0:
                    skip[i] = skip[i][:, :, :, -s.size(3):]
                skip[i] = s + skip[i]

            for i in range(self.n_mix):
                h = (self.models[i].gconv[0](x[i], self.adj_mxs[i]) +
                     self.models[i].gconv[1](x[i], self.adj_mxs[i].transpose(1, 0))) * (self.cross_weight[i, i] + 1)

                for j in range(self.n_mix):
                    g, w = graphs[i, j], self.cross_weight[i, j]
                    if i == j:
                        w = w + 1
                    _h = self.models[i].gconv[2](x[j], g)
                    h = h + _h * w
                x[i] = h

            for i in range(self.n_mix):
                x[i] = x[i] + residual[i][:, :, :, -x[i].size(3):]
                x[i] = self.models[i].bn[l](x[i])

        for i in range(self.n_mix):
            x[i] = F.relu(skip[i])
            x[i] = F.relu(self.models[i].end_conv_1(x[i]))
            x[i] = self.models[i].end_conv_2(x[i])
            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.n_nodes).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

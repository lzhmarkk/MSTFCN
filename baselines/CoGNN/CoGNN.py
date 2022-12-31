import torch
import torch.nn as nn
import torch.nn.functional as F
from ..MTGNN.MTGNN import dilated_inception, mixprop, LayerNorm


class graph_directed_sep_init(nn.Module):
    def __init__(self, l_matrix, k, dim, device, init_adj, alpha=3, static_feat=None):
        super(graph_directed_sep_init, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix) ** 2
        self.l_matrix = l_matrix
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.lin2 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            self.emb2 = nn.ModuleList()

            cur_i = 0
            initemb1, initemb2 = list(), list()
            for num in l_matrix:
                # m, p, n = torch.svd(init_adj[cur_i:cur_i + num, cur_i:cur_i + num])
                m, p, n = torch.svd(init_adj)
                cur_i = cur_i + num
                initemb1.append(torch.mm(m[:, :dim], torch.diag(p[:dim] ** 0.5)))
                initemb2.append(torch.mm(torch.diag(p[:dim] ** 0.5), n[:, :dim].t()).t())

            for i, n_sub1 in enumerate(l_matrix):
                for j, n_sub2 in enumerate(l_matrix):
                    emb1_ = nn.Embedding(n_sub1, dim)
                    emb1_.weight.data.copy_(initemb1[i])
                    emb2_ = nn.Embedding(n_sub2, dim)
                    emb2_.weight.data.copy_(initemb2[j])
                    self.emb1.append(emb1_)
                    self.emb2.append(emb2_)
                    lin1_ = nn.Linear(dim, dim)
                    lin1_.weight.data = torch.eye(dim)
                    lin1_.bias.data = torch.zeros(dim)
                    lin2_ = nn.Linear(dim, dim)
                    lin2_.weight.data = torch.eye(dim)
                    lin2_.bias.data = torch.zeros(dim)
                    self.lin1.append(lin1_)
                    self.lin2.append(lin2_)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            # embedding的初始化
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i * n_mod + j
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb2[index](idx2))
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = self.lin1[i](nodevec1[i])
            nodevec2[i] = self.lin2[i](nodevec2[i])
            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1, 0))
            adj_list.append(a)
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i * n_mod:(i + 1) * n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class CoGNN(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A,
                 dropout, subgraph_size, node_dim, dilation_exponential, conv_channels,
                 residual_channels, skip_channels, end_channels, window, horizon, in_dim, out_dim,
                 layers, propalpha, tanhalpha, layer_norm_affline=True):
        super(CoGNN, self).__init__()

        self.n_mix = len(in_dim)
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = torch.from_numpy(predefined_A).to(device)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.input_dim = in_dim
        self.output_dim = out_dim
        _dim = 0
        self.in_split, self.out_split = [], []
        for d in self.input_dim:
            self.in_split.append((_dim, _dim + d))
            _dim += d
        _dim = 0
        for d in self.output_dim:
            self.out_split.append((_dim, _dim + d))

        self.gc = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim[0],
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        # graph_directed_sep_init(l_matrix, subgraph_size, node_dim, device, predefined_A, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = window
        self.window = window
        self.horizon = horizon
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    if self.buildA_true:
                        self.gc.append(graph_directed_sep_init([self.num_nodes] * self.n_mix, subgraph_size, node_dim,
                                                               device, self.predefined_A,
                                                               alpha=tanhalpha, static_feat=None))
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes * self.n_mix, self.seq_length - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes * self.n_mix, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim[0] * horizon,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim[0], out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim[0], out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.idx = torch.arange(self.num_nodes * self.n_mix).to(device)

    def forward(self, input, **kwargs):
        input, time = input[..., :-2], input[..., -2:]
        input = [input[..., p[0]: p[1]] for p in self.in_split]  # (B, T, N, C_mix)
        input = torch.cat(input, dim=2)  # (B, T, N_mix, C)
        input = input.transpose(3, 1)

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            if self.gcn_true:
                if self.buildA_true:
                    adp = self.gc[i](self.idx)
                else:
                    adp = self.predefined_A
            residual = x
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # (B, T*C, N_mix, 1)
        h = []
        for i in range(self.n_mix):
            _h = x[:, :, i * self.num_nodes:(i + 1) * self.num_nodes, :]
            _h = _h.reshape(x.shape[0], self.horizon, self.output_dim[i], self.num_nodes)
            h.append(_h.permute(0, 1, 3, 2))
        h = torch.cat(h, dim=-1)  # (B, T, N, C_mix)
        return h

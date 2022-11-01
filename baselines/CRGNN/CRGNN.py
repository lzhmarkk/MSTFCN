from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(GCN, self).__init__()
        self.mlp = torch.nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def nconv(self, x, A):
        return torch.einsum('ncwl,vw->ncvl', (x, A)).contiguous()

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class CRGNN(nn.Module):
    def __init__(self, device, adj_mx, gcn_true, buildA_true, num_nodes, gcn_depth, dropout, input_dim, output_dim,
                 window, horizon, subgraph_size, node_dim, tanhalpha, propalpha, dilation_exponential,
                 layers, residual_channels, conv_channels, skip_channels, end_channels):
        super(CRGNN, self).__init__()
        self.device = device
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.gcn_depth = gcn_depth
        self.dropout = dropout
        self.predefined_A = (torch.tensor(adj_mx) - torch.eye(self.num_nodes)).to(self.device).float()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.in_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim
        self.residual_channels = residual_channels
        self.subgraph_size = subgraph_size
        self.node_dim = node_dim
        self.tanhalpha = tanhalpha
        self.propalpha = propalpha
        self.static_feat = None
        self.layer_norm_affline = True
        self.seq_length = window
        self.dilation_exponential = dilation_exponential
        self.layers = layers
        self.residual_channels = residual_channels
        self.conv_channels = conv_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels

        # modules
        self.idx = torch.arange(self.num_nodes).to(self.device)
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, self.node_dim, self.device,
                                    alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                    self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        if self.dilation_exponential > 1:
            rf_size_i = int(1 + 0 * (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                    self.dilation_exponential - 1))
        else:
            rf_size_i = 0 * self.layers * (kernel_size - 1) + 1

        # layers
        new_dilation = 1
        for j in range(1, self.layers + 1):
            if self.dilation_exponential > 1:
                rf_size_j = int(rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1) / (
                        self.dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
            self.gate_convs.append(
                dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
            self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                 out_channels=self.residual_channels,
                                                 kernel_size=(1, 1)))
            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            if self.gcn_true:
                self.gconv1.append(GCN(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                       self.propalpha))
                self.gconv2.append(GCN(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                       self.propalpha))

            if self.seq_length > self.receptive_field:
                self.norm.append(
                    LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),
                              elementwise_affine=self.layer_norm_affline))
            else:
                self.norm.append(
                    LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                              elementwise_affine=self.layer_norm_affline))

            new_dilation *= self.dilation_exponential

        # skip connection
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, 1), bias=True)

        # final output
        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=3 * self.skip_channels,
                                                out_channels=self.end_channels,
                                                kernel_size=(1, 1),
                                                bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=self.end_channels,
                                                out_channels=self.horizon * self.output_dim,
                                                kernel_size=(1, 1),
                                                bias=True))

    def forward(self, input, **kwargs):
        x = input[..., :self.in_dim]  # (bs, window, num_nodes, in_dim)
        time = input[..., self.in_dim:]  # (bs, window, num_nodes, 2)
        input = x.transpose(3, 1)  # (bs, in_dim, num_nodes, window)

        input = self.pad_sequence(input)

        g = self.build_graph(x)

        x = self.start_conv(input)  # (bs, res_channel, n_nodes, recep_field)
        output = self.skip0(F.dropout(input, self.dropout, training=self.training))  # (bs, skp_channel, n_nodes, 1)

        for i in range(self.layers):
            residual = x
            x = self.temporal_conv(x, i)

            _out = self.skip_convs[i](x)  # (bs, skp_channel, n_nodes, 1)
            output = _out + output

            x = self.spatial_conv(x, g, i)

            x = x + residual[:, :, :, -x.size(3):]  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
            x = self.norm[i](x, self.idx)

        output = self.skipE(x) + output
        x = F.relu(output)

        # time encoding
        x = self.time_encoding(x, time)
        x = self.end_conv(x)

        batch, size1, num_nodes, size3 = x.shape  # size3 = 1, size1 = horizon * out_dim
        x = x.reshape(batch, self.horizon, self.output_dim, num_nodes, size3).squeeze(dim=-1)
        x = x.permute(0, 1, 3, 2)
        return x

    def pad_sequence(self, x):
        if self.seq_length < self.receptive_field:
            return F.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))
        else:
            return x

    def build_graph(self, x):
        if self.gcn_true:
            if self.buildA_true:
                g = self.gc(self.idx)
            else:
                g = self.predefined_A
            return g

    def temporal_conv(self, x, layer):
        filter = torch.tanh(self.filter_convs[layer](x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
        gate = torch.sigmoid(self.gate_convs[layer](x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=self.training)  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
        return x

    def spatial_conv(self, x, g, layer):
        if self.gcn_true:
            x = self.gconv1[layer](x, g) + self.gconv2[layer](x, g.transpose(1, 0))
        else:
            x = x
        return x

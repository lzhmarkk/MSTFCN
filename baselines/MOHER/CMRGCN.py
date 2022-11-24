import torch
import torch.nn as nn


class CMRGCN(nn.Module):
    def __init__(self, device, dim, n_nodes, n_mix, n_layers, n_heads, n_relations, subgraph_size, summarize):
        super(CMRGCN, self).__init__()

        self.device = device
        self.dim = dim
        self.n_mix = n_mix
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_relations = n_relations
        self.n_nodes = n_nodes
        self.subgraph_size = subgraph_size
        self.summarize = summarize

        self.residual = nn.Linear(dim, dim, bias=True)
        # self.residual = nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), bias=True)

        self.a_weight = nn.Parameter(torch.randn(2, n_relations, n_heads, n_layers, 1, 1), requires_grad=True)
        self.B_weight = nn.Parameter(torch.randn(1, n_heads, n_layers, dim, dim), requires_grad=True)
        self.a_bias = nn.Parameter(torch.randn(2, n_relations, n_heads, n_layers, 1), requires_grad=True)
        self.B_bias = nn.Parameter(torch.randn(1, n_heads, n_layers, dim), requires_grad=True)

    def get_parameters(self):
        # c
        weight_c = self.a_weight[0] * self.B_weight  # (n_rel, n_heads, n_layers, dim, dim)
        weight_c = weight_c.sum(dim=1).contiguous()  # (n_rel, n_layers, dim, dim)
        bias_c = self.a_bias[0] * self.B_bias  # (n_rel, n_heads, n_layers, dim)
        bias_c = bias_c.sum(dim=1).contiguous()  # (n_rel, n_layers, dim)
        # d
        weight_d = self.a_weight[1] * self.B_weight  # (n_rel, n_heads, n_layers, dim, dim)
        weight_d = weight_d.sum(dim=1).contiguous()  # (n_rel, n_layers, dim, dim)
        bias_d = self.a_bias[1] * self.B_bias  # (n_rel, n_heads, n_layers, dim)
        bias_d = bias_d.sum(dim=1).contiguous()  # (n_rel, n_layers, dim)
        return weight_c, bias_c, weight_d, bias_d

    def CMRGCNLayer(self, x, graphs, weight_c, bias_c, weight_d, bias_d):
        # x: (bs, window, n_nodes, dim) * n_mix
        # graphs: (n_nodes, n_nodes) * n_type
        for i in range(self.n_mix):
            x[i] = x[i].transpose(3, 2)

        h = [0.] * self.n_mix
        for i in range(self.n_mix):
            for t, adj in enumerate(graphs):
                for j in range(self.n_mix):
                    rel = (t * self.n_mix + i) * self.n_mix + j
                    c = torch.matmul(x[j], adj)
                    d = torch.matmul(x[j] - x[i], adj)
                    c = c.transpose(3, 2)
                    d = d.transpose(3, 2)
                    c = torch.relu(torch.matmul(c, weight_c[rel]) + bias_c[rel])
                    d = torch.tanh(torch.matmul(d, weight_d[rel]) + bias_d[rel])
                    h[i] = h[i] + c + d

        for i in range(self.n_mix):
            x[i] = h[i]
        return x

    def forward(self, x, graph, neighbors, neighbors_weight):
        # x: (bs, dim, n_nodes, window) * n_mix
        g = [[x[i]] for i in range(self.n_mix)]
        for i in range(self.n_mix):
            x[i] = x[i].transpose(3, 1)

        weight_c, bias_c, weight_d, bias_d = self.get_parameters()

        for l in range(self.n_layers):
            x = self.CMRGCNLayer(x, graph, weight_c[:, l], bias_c[:, l], weight_d[:, l], bias_d[:, l])
            for i in range(self.n_mix):
                g[i].append(x[i].transpose(3, 1))

        for i in range(self.n_mix):
            g[i] = torch.cat(g[i], dim=1)  # (bs, n_layers * dim, n_nodes, window) * n_mix

        # summarizing
        if self.summarize:
            for i in range(self.n_mix):
                __ = []
                for t, (adj, nei, wei) in enumerate(zip(graph, neighbors, neighbors_weight)):
                    _ = []
                    for tgt in range(self.n_nodes):
                        h = g[i][:, :, nei[tgt], :] * wei[tgt].reshape(1, 1, len(wei[tgt]), 1)
                        _.append(torch.sum(h, 2))  # (bs, n_layers * dim, window)
                    _ = torch.stack(_, 2)  # (bs, n_layers * dim, n_nodes, window)
                    __.append(_)
                g[i] = torch.cat(__, 1)

        return g

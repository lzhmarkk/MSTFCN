import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fn


class STSHN(nn.Module):
    """
    https://github.com/akaxlh/ST-SHN
    pytorch version implemented by @lzhmark, 20221205
    """

    def __init__(self, device, input_dim, output_dim, num_nodes, window, horizon, spatial_layers, temporal_layers,
                 embed_dim, adj, heads, dropout, hyper_num):
        super(STSHN, self).__init__()

        self.device = device
        self.n_mix = len(input_dim)
        self.n_categories = sum(input_dim)
        assert sum(input_dim) == sum(output_dim)
        self.n_nodes = num_nodes
        self.embed_dim = embed_dim
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        self.window = window
        self.horizon = horizon

        self.categories_emb = nn.Embedding(self.n_categories, embed_dim)

        self.adj = adj.float().to(self.device)
        self.hyper_adj = nn.Parameter(torch.randn([hyper_num, num_nodes]), requires_grad=True)

        self.heads = heads
        self.sQ = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)
        self.sK = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)
        self.sV = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)

        self.tQ = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)
        self.tK = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)
        self.tV = nn.ModuleList([nn.Linear(embed_dim, embed_dim // self.heads, bias=False)] * self.heads)

        self.w = nn.Parameter(torch.randn([1, 1, self.n_categories, embed_dim]), requires_grad=True)
        self.output = nn.Linear(embed_dim, horizon, bias=True)

        self.dropout = nn.Dropout(dropout)

        self.to(self.device)

    def spatial_modelling(self, x):
        h = []
        for i in range(self.heads):
            q = self.sQ[i](x.unsqueeze(4))  # (B, N, T, C, 1, D/heads)
            k = self.sK[i](x.unsqueeze(3))  # (B, N, T, 1, C, D/heads)
            v = self.sV[i](x.unsqueeze(3))
            a = torch.sum(q * k, dim=-1) / np.sqrt(self.embed_dim // self.heads)  # (B, N, T, C, C)
            a = fn.softmax(a, dim=4).unsqueeze(-1)  # (B, N, T, C, C, 1)
            v = torch.sum(a * v, dim=4)  # (B, N, T, C, D/heads)
            h.append(v)
        h = torch.cat(h, dim=-1)  # (B, N, T, C, D)

        h = torch.einsum("bntcd,nm->bmtcd", h, self.dropout(self.adj))
        h = fn.leaky_relu(h)
        return h

    def hyperGNN(self, x):
        # x (B, N, T, C, D)
        tp_adj = self.hyper_adj.transpose(1, 0)
        hyper_embeds = fn.leaky_relu(torch.einsum('mn,bntcd->bmtcd', self.dropout(self.hyper_adj), x))
        ret_embeds = fn.leaky_relu(torch.einsum('nm,bmtcd->bntcd', self.dropout(tp_adj), hyper_embeds))
        return ret_embeds

    def temporal_modelling(self, x):
        nextTEmbeds = x[:, :, 1:, :, :]  # (B, N, T-1, C, D)
        prevTEmbeds = x[:, :, :-1, :, :]  # (B, N, T-1, C, D)

        h = []
        for i in range(self.heads):
            q = self.tQ[i](nextTEmbeds.unsqueeze(4))  # (B, N, T-1, C, 1, D/heads)
            k = self.tK[i](prevTEmbeds.unsqueeze(3))  # (B, N, T-1, 1, C, D/heads)
            v = self.tV[i](prevTEmbeds.unsqueeze(3))
            a = torch.sum(q * k, dim=-1) / np.sqrt(self.embed_dim // self.heads)  # (B, N, T-1, C, C)
            a = fn.softmax(a, dim=4).unsqueeze(-1)  # (B, N, T-1, C, C, 1)
            v = torch.sum(a * v, dim=4)  # (B, N, T-1, C, D/heads)
            h.append(v)
        h = torch.cat(h, dim=-1)  # (B, N, T-1, C, D)
        h = torch.cat([x[:, :, [0], :, :], h], dim=2)
        return fn.leaky_relu(h)

    def forward(self, input, **kwargs):
        input, time = input[..., :-2], input[..., -2:]
        # input (B, T, N, C, 1)
        x = input.transpose(2, 1).unsqueeze(-1)
        # (B, N, T, C, D)
        initialEmbeds = x * self.categories_emb.weight.reshape(1, 1, 1, self.n_categories, self.embed_dim)

        embed = initialEmbeds
        embeds = initialEmbeds
        for _ in range(self.spatial_layers):
            embed = self.spatial_modelling(embed)
            embed = self.hyperGNN(embed) + embed
            embeds += embed
        embed = embeds / self.spatial_layers  # (B, N, T, C, D)

        embeds = embed
        for _ in range(self.temporal_layers):
            embed = self.temporal_modelling(embed)
            embeds += embed
        embed = embeds / self.temporal_layers  # (B, N, T, C, D)

        embed = torch.mean(embed, dim=2)  # (B, N, C, D)
        y = embed * self.w  # (B, N, C, D)
        y = self.output(y).permute(0, 3, 1, 2)  # (B, T, N, C)
        return y

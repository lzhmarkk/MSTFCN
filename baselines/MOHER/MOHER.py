import torch
import torch.nn as nn
from .GraphConstructor import GraphConstructor
from .CMRGCN import CMRGCN
from ..CRGNN.TimeEncoder import TimeEncoder


class MOHER(nn.Module):
    """
    Q. Zhou et al., “Modeling Heterogeneous Relations across Multiple Modes for Potential Crowd Flow Prediction,”
    in Proceedings of the AAAI Conference on Artificial Intelligence, May 2021, vol. 35, pp. 4723–4731.
    doi: 10.1609/aaai.v35i5.16603.

    Implement by lzhmark, 20221107
    """

    def __init__(self, device, adj_mx, num_nodes, window, horizon, input_dim, output_dim, gamma, beta, subgraph_size,
                 static_feat, n_heads, n_layers, hidden_dim, dropout, summarize, add_time):
        super(MOHER, self).__init__()

        self.adj_mx = adj_mx
        self.num_nodes = num_nodes
        self.window = window
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_mix = len(input_dim)
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.subgraph_size = subgraph_size
        self.static_feat = static_feat
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_poi_graph = self.static_feat is not None
        self.dropout = dropout
        self.summarize = summarize
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

        graph_constructor = GraphConstructor(device=device, n_mix=1, n_nodes=self.num_nodes,
                                             adj_mx=self.adj_mx,
                                             gamma=self.gamma, beta=self.beta, subgraph_size=self.subgraph_size,
                                             poi_feat=self.static_feat)
        self.graphs, self.neighbors, self.neighbors_w = graph_constructor.get_graphs()
        self.n_rel = len(self.graphs) * self.n_mix * self.n_mix

        self.GCN = CMRGCN(device=device, dim=hidden_dim, n_layers=n_layers, n_mix=self.n_mix, n_heads=n_heads,
                          n_relations=self.n_rel, n_nodes=num_nodes, subgraph_size=subgraph_size, summarize=summarize)

        self.start_conv = nn.ModuleList([nn.Conv2d(self.input_dim[i], hidden_dim, kernel_size=(1, 1))
                                         for i in range(self.n_mix)])

        self.lstm = nn.ModuleList([nn.LSTM(input_size=(1 + self.n_layers) * hidden_dim,
                                           hidden_size=hidden_dim,
                                           num_layers=2,
                                           batch_first=True)] * self.n_mix)

        self.predict_conv = nn.ModuleList([nn.Linear(window, horizon, bias=True)] * self.n_mix)

        self.end_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_dim * (3 if self.add_time else 1), 128, kernel_size=(1, 1), bias=True),
                nn.ReLU(),
                nn.Conv2d(128, self.horizon * self.output_dim[i], kernel_size=(1, 1), bias=True))
            for i in range(self.n_mix)])

        if self.add_time:
            self.time_encoder = TimeEncoder(dim=hidden_dim, length=window)

        self.to(self.device)

    def forward(self, input, **kwargs):
        input, time = input[..., :-2], input[..., -2:]
        bs = input.shape[0]
        x = input.transpose(3, 1)
        # x = self.pad_sequence(x)  # (bs, mix_in_dim, n_nodes, window)
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (bs, in_dim, n_nodes, window) * n_mix

        for i in range(self.n_mix):
            x[i] = self.start_conv[i](x[i])

        x = self.GCN(x, self.graphs, self.neighbors, self.neighbors_w)  # (bs, n_layers * dim, n_nodes, window) * n_mix

        # LSTM
        for i in range(self.n_mix):
            # (bs * n_nodes, window, (1+n_layers) * dim)
            x[i] = x[i].permute(0, 2, 3, 1).reshape(bs * self.num_nodes, self.window, -1)
            x[i], _ = self.lstm[i](x[i])  # (bs * n_nodes, window, dim)
            x[i] = x[i].sum(dim=1).reshape(bs, self.num_nodes, self.hidden_dim)  # (bs, n_nodes, dim)
            x[i] = x[i].permute(0, 2, 1).unsqueeze(dim=-1)  # (bs, dim, n_nodes, 1)
            x[i] = self.time_encoder(x[i], time) if self.add_time else x[i]  # (bs, dim, n_nodes, 1)
            x[i] = self.end_conv[i](x[i]).squeeze(dim=-1)  # (bs, horizon * output_dim, n_nodes)
            x[i] = x[i].reshape(bs, self.horizon, self.output_dim[i], self.num_nodes).permute(0, 1, 3, 2)

        x = torch.cat(x, -1)
        return x

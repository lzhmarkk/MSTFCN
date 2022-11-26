import torch
import torch.nn as nn


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
        for i, nodevec1 in enumerate(self.emb1):  # todo 反对称graph，两个方向的流量不同
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

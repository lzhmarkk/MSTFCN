import torch
import torch.nn as nn


class CrossRelationGraphConstructor(nn.Module):
    def __init__(self, n_mix, nnodes, k, dim, device, alpha=3, cross_relation=1, full_graph=False):
        """
        cross_relation
        0: no cross relation
        1: MTGNN style, M_{1,in}=MLP1(E1), M_{2,out}=MLP2(E2), A=M_{1,in}*M_{2,out}-M_{2,out}*M_{1,in}
        2: semantic style, M_{1,in}=MLP1(E1), M_{1,out}=MLP2(E1), M_{2,in}=MLP1(E2), M_{2,out}=MLP2(E2),
                        A=M_{1,in}*M_{2,out}-M_{1,out}*M_{2,in}
        """
        super(CrossRelationGraphConstructor, self).__init__()
        self.n_mix = n_mix
        self.n_nodes = nnodes
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.full_graph = full_graph
        self.cross = cross_relation
        print(f'Use {["no", "mtgnn", "semantic"][cross_relation]} cross relation')

        # use different mix emb
        self.emb = nn.ModuleList([nn.Embedding(self.n_nodes, self.dim)] * self.n_mix)
        # use mlp to identify mix emb
        # self.emb2 = nn.Embedding(self.n_nodes, self.dim)
        # self.emb2_mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=True)), nn.ReLU()] * self.n_mix)

        self.mlp_inflow = nn.Linear(dim, dim, bias=False)
        self.mlp_outflow = nn.Linear(dim, dim, bias=False)

    def __gen_graph(self, emb1, emb2):
        m1in = torch.tanh(self.alpha * self.mlp_inflow(emb1))
        m2out = torch.tanh(self.alpha * self.mlp_outflow(emb2))

        if self.cross == 2:
            m1out = torch.tanh(self.alpha * self.mlp_outflow(emb1))
            m2in = torch.tanh(self.alpha * self.mlp_inflow(emb2))
            a = torch.mm(m1in, m2out.transpose(1, 0)) - torch.mm(m1out, m2in.transpose(1, 0))
        else:
            a = torch.mm(m1in, m2out.transpose(1, 0)) - torch.mm(m2out, m1in.transpose(1, 0))

        adj = torch.relu(torch.tanh(self.alpha * a))

        if not self.full_graph:
            mask = torch.zeros(self.n_nodes, self.n_nodes).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

        return adj

    def forward(self):
        adjs = []
        for i, emb1 in enumerate(self.emb):
            adjs_row = []
            for j, emb2 in enumerate(self.emb):
                if self.cross == 0 and i != j:
                    adj = torch.zeros(self.n_nodes, self.n_nodes).to(self.device)
                else:
                    adj = self.__gen_graph(emb1.weight, emb2.weight)
                adjs_row.append(adj)  # (n_nodes, n_nodes)

            adjs.append(torch.stack(adjs_row, 0))  # (n_mix, n_nodes, n_nodes)
        return torch.stack(adjs, 0)  # (n_mix, n_mix, n_nodes, n_nodes)

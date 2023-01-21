import torch
import torch.nn as nn


class CSP(nn.Module):
    def __init__(self, c_in, c_out, gcn_depth, dropout, gcn_agg_func='mean'):
        super(CSP, self).__init__()

        self.gcn_agg_func = gcn_agg_func
        self.gcn_depth = gcn_depth

        if self.gcn_agg_func == 'mlp':
            self.mlp = nn.Conv2d((gcn_depth + 1) * c_in, c_out,
                                 kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        elif self.gcn_agg_func == 'sum':
            pass
        elif self.gcn_agg_func == 'mean':
            pass
        else:
            raise ValueError()

    def nconv(self, x, A):
        return torch.einsum('ncwl,vw->ncvl', (x, A)).contiguous()

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        if self.gcn_agg_func == 'mlp':
            for i in range(self.gcn_depth):
                h = self.nconv(h, a)
                out.append(h)

            ho = torch.cat(out, dim=1)
            ho = self.mlp(ho)
        elif self.gcn_agg_func == 'mean' or self.gcn_agg_func == 'sum':
            ap = [torch.eye(len(a)).to(a.device)]
            for i in range(self.gcn_depth):
                ap.append(torch.matmul(ap[-1], a))
            ho = self.nconv(x, torch.stack(ap, 0).sum(0))
            if self.gcn_agg_func == 'mean':
                ho = ho / (self.gcn_depth + 1)
        else:
            raise ValueError()
        return ho

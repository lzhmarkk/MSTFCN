import torch
import torch.nn as nn


class SpatialMixer(nn.Module):
    def __init__(self, c_in, c_out, gcn_depth, dropout, alpha, gcn_agg_func='mean'):
        super(SpatialMixer, self).__init__()

        self.alpha = alpha
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
        for i in range(self.gcn_depth):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)

        if self.gcn_agg_func == 'mlp':
            ho = torch.cat(out, dim=1)
            ho = self.mlp(ho)
        elif self.gcn_agg_func == 'sum':
            ho = torch.stack(out, dim=0)  # (gdep+1, B, C, N, T)
            ho = torch.sum(ho, dim=0)
        elif self.gcn_agg_func == 'mean':
            ho = torch.stack(out, dim=0)  # (gdep+1, B, C, N, T)
            ho = torch.mean(ho, dim=0)
        else:
            raise ValueError()
        return ho

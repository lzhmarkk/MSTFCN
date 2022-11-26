import torch
import torch.nn as nn


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
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


class TemporalMixer(nn.Module):
    def __init__(self, device, temporal_func, residual_channels, conv_channels, dilation, begin_dim, end_dim, dropout):
        super(TemporalMixer, self).__init__()
        self.device = device
        self.temporal_func = temporal_func

        if self.temporal_func == 'TCN':
            self.filter_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=dilation)
            self.gate_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=dilation)
            self.dropout = nn.Dropout(dropout)
        elif self.temporal_func == 'MLP':
            self.mlp = nn.Sequential(
                nn.Linear(begin_dim, end_dim, bias=True),
                nn.GELU(),
                nn.Linear(end_dim, end_dim, bias=True)
            )
        else:
            raise ValueError()

    def forward(self, x):
        if self.temporal_func == 'TCN':
            filter = torch.tanh(self.filter_conv(x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
            gate = torch.sigmoid(self.gate_conv(x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
            x = filter * gate
            x = self.dropout(x)
        elif self.temporal_func == 'MLP':
            x = self.mlp(x)
        else:
            raise ValueError()
        return x

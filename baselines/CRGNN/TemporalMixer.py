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
    def __init__(self, device, temporal_func, residual_channels, conv_channels, begin_dim, end_dim, dropout):
        super(TemporalMixer, self).__init__()
        self.device = device
        self.temporal_func = temporal_func

        if self.temporal_func == 'TCN':
            self.filter_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=1)
            self.gate_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=1)
            self.dropout = nn.Dropout(dropout)
        elif self.temporal_func == 'GRU':
            self.gru = nn.GRU(conv_channels, conv_channels, 2, batch_first=True, bidirectional=True)
        elif self.temporal_func == 'Attention':
            attn_layer = nn.TransformerEncoderLayer(d_model=residual_channels, nhead=4, dropout=0.,
                                                    dim_feedforward=2 * residual_channels, batch_first=True)
            self.attn = nn.TransformerEncoder(encoder_layer=attn_layer, num_layers=4)
        elif self.temporal_func == 'MLP':
            self.mlp = nn.Sequential(
                nn.Linear(begin_dim, end_dim, bias=True),
                nn.GELU(),
                nn.Linear(end_dim, end_dim, bias=True)
            )
        else:
            raise ValueError()
        self.end_dim = end_dim

    def forward(self, x):
        if self.temporal_func == 'TCN':
            filter = torch.tanh(self.filter_conv(x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
            gate = torch.sigmoid(self.gate_conv(x))  # (bs, 32, n_nodes, recep_filed + (1 - max_ker_size) * i)
            x = filter * gate
            x = x[..., -self.end_dim:]
        elif self.temporal_func == 'GRU':
            B, C, N, T = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * N, T, C)  # (B*N, T, C)
            output, hidden = self.gru(x)
            x = output.reshape(B, N, T, 2, C)[..., 0, :].permute(0, 3, 1, 2)  # (B, C, N, T)
            x = x[..., -self.end_dim:]
        elif self.temporal_func == 'Attention':
            B, C, N, T = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * N, T, C)  # (B*N, T, C)
            output = self.attn(x)
            x = output.reshape(B, N, T, C).permute(0, 3, 1, 2)  # (B, C, N, T)
            x = x[..., -self.end_dim:]
        elif self.temporal_func == 'MLP':
            x = self.mlp(x)
        else:
            raise ValueError()
        return x

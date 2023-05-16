import torch
import torch.nn as nn
import torch.nn.functional as fn
from baselines.MTGNN.MTGNN import dilated_inception
from baselines.MTGNN.MTGNN import mixprop


class FTLayer(nn.Module):
    def __init__(self, n_dim, n_nodes, start_length, end_length, temporal_func, frequency_func, spatial_func, channel_func, dropout):
        super().__init__()

        self.n_dim = n_dim
        self.n_nodes = n_nodes
        self.start_length = start_length
        self.end_length = end_length
        self.temporal_func = temporal_func
        self.spatial_func = spatial_func
        self.frequency_func = frequency_func
        self.channel_func = channel_func
        self.dropout = nn.Dropout(dropout)

        # temporal
        if temporal_func == 'FC':
            self.t_fc = nn.Linear(start_length, end_length)
        elif temporal_func == 'MLP':
            self.t_mlp = nn.Sequential(nn.Linear(start_length, end_length),
                                       nn.GELU(),
                                       nn.Linear(end_length, end_length))
        elif temporal_func == 'TCN':
            self.filter_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)
            self.gate_convs = dilated_inception(n_dim, n_dim, dilation_factor=1)
        elif temporal_func == 'FFTW':
            self.temporal_weight = nn.Parameter(torch.randn((n_dim, n_dim, start_length // 2 + 1), dtype=torch.cfloat))
            self.temporal_bias = nn.Parameter(torch.randn((1, n_dim, 1, start_length // 2 + 1), dtype=torch.cfloat))
        elif temporal_func == 'FFTN':
            self.temporal_weight1 = nn.Parameter(torch.randn((n_dim, 2 * n_dim, start_length // 2 + 1), dtype=torch.cfloat))
            self.temporal_bias1 = nn.Parameter(torch.randn((1, 2 * n_dim, 1, start_length // 2 + 1), dtype=torch.cfloat))
            self.temporal_weight2 = nn.Parameter(torch.randn((2 * n_dim, n_dim, start_length // 2 + 1), dtype=torch.cfloat))
            self.temporal_bias2 = nn.Parameter(torch.randn((1, n_dim, 1, start_length // 2 + 1), dtype=torch.cfloat))

        # spatial
        self.gconv1 = mixprop(n_dim, n_dim, 1, dropout, 0.00)
        self.gconv2 = mixprop(n_dim, n_dim, 1, dropout, 0.00)

        # frequency
        if frequency_func == 'FC':
            self.frequency_weight = nn.Parameter(torch.randn((n_dim, start_length // 2 + 1, start_length // 2 + 1), dtype=torch.cfloat))
            self.frequency_bias = nn.Parameter(torch.randn((1, 1, 1, start_length // 2 + 1), dtype=torch.cfloat))
        elif frequency_func == 'FFN':
            self.frequency_weight1 = nn.Parameter(torch.randn((n_dim, start_length // 2 + 1, start_length), dtype=torch.cfloat))
            self.frequency_bias1 = nn.Parameter(torch.randn((1, 1, 1, start_length), dtype=torch.cfloat))
            self.frequency_weight2 = nn.Parameter(torch.randn((n_dim, start_length, start_length // 2 + 1), dtype=torch.cfloat))
            self.frequency_bias2 = nn.Parameter(torch.randn((1, 1, 1, start_length // 2 + 1), dtype=torch.cfloat))

        # channel
        if channel_func == 'FC':
            self.channel_weight = nn.Parameter(torch.randn((n_dim, n_dim), dtype=torch.cfloat))
            self.channel_bias = nn.Parameter(torch.randn((1, n_dim, 1, 1), dtype=torch.cfloat))
        elif channel_func == 'FFN':
            self.channel_weight1 = nn.Parameter(torch.randn((n_dim, 2 * n_dim), dtype=torch.cfloat))
            self.channel_bias1 = nn.Parameter(torch.randn((1, 2 * n_dim, 1, 1), dtype=torch.cfloat))
            self.channel_weight2 = nn.Parameter(torch.randn((2 * n_dim, n_dim), dtype=torch.cfloat))
            self.channel_bias2 = nn.Parameter(torch.randn((1, n_dim, 1, 1), dtype=torch.cfloat))

        self.ln = nn.LayerNorm([n_dim, n_nodes, end_length], elementwise_affine=True)

    def temporal_mixing(self, x):
        res = x
        assert self.temporal_func == 'FFT'

        if self.temporal_func == 'FC':
            x = torch.tanh(self.t_fc(x))
        elif self.temporal_func == 'MLP':
            x = torch.tanh(self.t_mlp(x))
        elif self.temporal_func == 'TCN':
            filter = torch.tanh(self.filter_convs(x))
            gate = torch.sigmoid(self.gate_convs(x))
            x = filter * gate
        elif self.temporal_func == 'FFTW':
            x = torch.einsum('bcnt,cdt->bdnt', x, self.temporal_weight)
            x = x + self.temporal_bias
        elif self.temporal_func == 'FFTN':
            x = torch.einsum('bcnt,cdt->bdnt', x, self.temporal_weight1)
            x = x + self.temporal_bias1
            x = fn.gelu(x.real) + 1.j * fn.gelu(x.imag)
            x = torch.einsum('bcnt,cdt->bdnt', x, self.temporal_weight2)
            x = x + self.temporal_bias2
        else:
            x = torch.zeros_like(x)

        x = x + res
        return x

    def frequency_mixing(self, x):
        res = x

        if self.frequency_func == 'FC':
            assert torch.is_complex(x)
            x = torch.einsum('bcnt,ctf->bcnf', x, self.frequency_weight)
            x = x + self.frequency_bias
        elif self.frequency_func == 'FFN':
            assert torch.is_complex(x)
            x = torch.einsum('bcnt,ctf->bcnf', x, self.frequency_weight1)
            x = x + self.frequency_bias1
            x = fn.gelu(x.real) + 1.j * fn.gelu(x.imag)
            x = torch.einsum('bcnt,ctf->bcnf', x, self.frequency_weight2)
            x = x + self.frequency_bias2
        else:
            x = torch.zeros_like(x)

        x = x + res
        return x

    def spatial_mixing(self, x, g):  # todo dynamic graph
        res = x

        if torch.is_complex(x):
            real = self.gconv1(x.real, g[0]) + self.gconv2(x.real, g[0].transpose(1, 0))
            imag = self.gconv1(x.imag, g[1]) + self.gconv2(x.imag, g[1].transpose(1, 0))
            x = fn.tanh(real) + 1.j * fn.tanh(imag)

        else:
            x = self.gconv1(x, g) + self.gconv2(x, g.transpose(1, 0))

        x = x + res
        return x

    def channel_mixing(self, x):
        res = x

        if self.channel_func == 'FC':
            x = torch.einsum("bcnt,cd->bdnt", x, self.channel_weight)
            x = x + self.channel_bias
        elif self.channel_func == 'FFN':
            x = torch.einsum("bcnt,cd->bdnt", x, self.channel_weight1)
            x = x + self.channel_bias1
            x = fn.leaky_relu(x.real) + 1.j * fn.leaky_relu(x.imag)
            x = torch.einsum("bcnt,cd->bdnt", x, self.channel_weight2)
            x = x + self.channel_bias2
        else:
            x = torch.zeros_like(x)

        x = x + res
        return x

    def forward(self, x, g):
        # (B, C, N, T), (N, N)
        residual = x

        if 'FFT' in self.temporal_func:
            x = torch.fft.rfft(x, dim=3, norm='ortho')

        x = self.temporal_mixing(x)

        x = self.frequency_mixing(x)

        x = self.spatial_mixing(x, g)

        x = self.channel_mixing(x)

        if 'FFT' in self.temporal_func:  # todo do not remap to time domain
            x = torch.fft.irfft(x, n=self.end_length, dim=3, norm='ortho')
            x = torch.tanh(x)

        # residual
        # x = x + residual[..., -self.end_length:]
        # x = x + self.res_conv(residual[..., -self.end_length:])
        x = self.ln(x)
        return x

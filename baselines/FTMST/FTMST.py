import torch
import torch.nn as nn
import torch.nn.functional as fn
from .layer import FTLayer


class FTMST(nn.Module):
    def __init__(self, n_dim, n_layer, n_nodes, input_dim, output_dim, window, horizon,
                 temporal_func, spatial_func, dropout, add_time):
        super().__init__()

        if not isinstance(input_dim, list):
            input_dim = [input_dim]
            output_dim = [output_dim]
        self.n_mix = len(input_dim)
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.add_time = add_time
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window = window
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.temporal_func = temporal_func
        self.spatial_func = spatial_func
        self.dropout = nn.Dropout(p=dropout)
        self.handle_seq_length()

        _dim = 0
        self.in_split, self.out_split = [], []
        for d in self.input_dim:
            self.in_split.append((_dim, _dim + d))
            _dim += d
        _dim = 0
        for d in self.output_dim:
            self.out_split.append((_dim, _dim + d))
            _dim += d

        self.init_conv = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.agg_conv = nn.ModuleList()
        self.pred_conv = nn.ModuleList()
        for i in range(self.n_mix):
            self.init_conv.append(nn.Conv2d(input_dim[i], n_dim, kernel_size=(1, 1), bias=True))
            out_conv_mode = nn.ModuleList()
            layers_mode = nn.ModuleList()

            out_conv_mode.append(nn.Conv2d(n_dim, n_dim, kernel_size=(1, self.start_length[0]), bias=True))
            for l in range(self.n_layer):
                out_conv_mode.append(nn.Conv2d(n_dim, n_dim, kernel_size=(1, self.end_length[l]), bias=True))
                layers_mode.append(FTLayer(n_dim, n_nodes, self.start_length[l], self.end_length[l],
                                           temporal_func, spatial_func, dropout))

            self.out_conv.append(out_conv_mode)
            self.layers.append(layers_mode)

            self.agg_conv.append(nn.Conv2d(n_dim * (self.n_layer + 1), n_dim, kernel_size=(1, 1), bias=True))
            self.pred_conv.append(nn.Sequential(nn.Conv2d(n_dim, 4 * n_dim, kernel_size=(1, 1), bias=True),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(4 * n_dim, horizon * output_dim[i], kernel_size=(1, 1), bias=True)))

    def handle_seq_length(self):
        self.start_length, self.end_length = [], []
        for l in range(self.n_layer):
            if self.temporal_func == 'TCN':
                self.start_length.append((self.n_layer - l) * 6 + 1)
                self.end_length.append((self.n_layer - l) * 6 - 5)
            elif self.temporal_func in ['MLP', 'FC', 'FFT']:
                self.start_length.append(self.window - self.window // self.n_layer * l)
                self.end_length.append(max(self.window - self.window // self.n_layer * (l + 1), 1))
            else:
                self.start_length.append(self.window)
                self.end_length.append(self.window)

    def forward(self, input, **kwargs):
        if self.add_time:
            x, time = input[..., :-2], input[..., -2:]
        else:
            x = input

        x = x.transpose(3, 1)  # (B, M*C, N, T)
        x = fn.pad(x, [self.start_length[0] - x.shape[-1], 0])
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (M, B, C, N, T)

        outputs = []

        for m in range(self.n_mix):
            x[m] = self.init_conv[m](x[m])

            h = self.dropout(self.out_conv[m][0](x[m]))  # (B, C, N, 1)
            outputs.append([h])

        for m in range(self.n_mix):
            for l in range(self.n_layer):
                residual = x[m]
                h = self.layers[m][l](x[m])  # (B, C, N, T)

                # residual
                h = h + residual[..., -h.shape[-1]:]
                h = self.layers[m][l].norm(h)

                x[m] = h
                _h = self.out_conv[m][l + 1](h)
                _h = self.dropout(_h)
                outputs[m].append(_h)

            outputs[m] = torch.cat(outputs[m], dim=1)
            outputs[m] = self.agg_conv[m](outputs[m])

        # output
        for m in range(self.n_mix):
            outputs[m] = fn.leaky_relu(outputs[m])
            h = self.pred_conv[m](outputs[m])  # (B, T*C, N, 1)
            outputs[m] = h.reshape(-1, self.horizon, self.output_dim[m], self.n_nodes).transpose(3, 2)

        y = torch.cat(outputs, -1)
        return y

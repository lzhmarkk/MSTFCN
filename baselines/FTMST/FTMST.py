import torch
import torch.nn as nn
import torch.nn.functional as fn
from .layer import FTLayer


class FTMST(nn.Module):
    def __init__(self, n_dim, n_layer, n_nodes, input_dim, output_dim, window, horizon, dropout, add_time):
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

        _dim = 0
        self.in_split, self.out_split = [], []
        for d in self.input_dim:
            self.in_split.append((_dim, _dim + d))
            _dim += d
        _dim = 0
        for d in self.output_dim:
            self.out_split.append((_dim, _dim + d))
            _dim += d

        self.dropout = nn.Dropout(p=dropout)

        self.init_conv = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.agg_conv = nn.ModuleList()
        self.pred_conv = nn.ModuleList()
        for i in range(self.n_mix):
            self.init_conv.append(nn.Conv2d(input_dim[i], n_dim, kernel_size=(1, 1), bias=True))
            out_conv_mode = nn.ModuleList()
            layers_mode = nn.ModuleList()

            out_conv_mode.append(nn.Conv2d(n_dim, n_dim, kernel_size=(1, window), bias=True))
            for l in range(self.n_layer):
                start_length = window - window // n_layer * l
                end_length = max(window - window // n_layer * (l + 1), 1)
                out_conv_mode.append(nn.Conv2d(n_dim, n_dim, kernel_size=(1, end_length), bias=True))
                layers_mode.append(FTLayer(n_dim, n_nodes, start_length, end_length, dropout))

            self.out_conv.append(out_conv_mode)
            self.layers.append(layers_mode)

            self.agg_conv.append(nn.Conv2d(n_dim * (self.n_layer + 1), n_dim, kernel_size=(1, 1), bias=True))
            self.pred_conv.append(nn.Sequential(nn.Conv2d(n_dim, 4 * n_dim, kernel_size=(1, 1), bias=True),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(4 * n_dim, horizon * output_dim[i], kernel_size=(1, 1), bias=True)))

    def forward(self, input, **kwargs):
        if self.add_time:
            x, time = input[..., :-2], input[..., -2:]
        else:
            x = input

        x = x.transpose(3, 1)  # (B, M*C, N, T)
        x = [x[:, p[0]: p[1]] for p in self.in_split]  # (M, B, C, N, T)
        # x[0] = fn.pad(x[0], [7, 0])

        # todo add your model here
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
        # todo skip e
        for m in range(self.n_mix):
            outputs[m] = fn.leaky_relu(outputs[m])
            h = self.pred_conv[m](outputs[m])  # (B, T*C, N, 1)
            outputs[m] = h.reshape(-1, self.horizon, self.output_dim[m], self.n_nodes).transpose(3, 2)

        y = torch.cat(outputs, -1)
        return y

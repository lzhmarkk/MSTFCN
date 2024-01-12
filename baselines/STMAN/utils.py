import math
import torch
import torch.nn as nn


class conv2d(nn.Module):
    def __init__(self, input_dim, output_dim, activation, bn, use_bias=True):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, [1, 1], [1, 1], bias=use_bias)
        self.activation = activation
        if self.activation is not None:
            self.bn = nn.LayerNorm(output_dim)  # nn.BatchNorm2d(output_dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # use conv 1x1 to emulate Linear
        if self.activation is not None:
            if self.bn:
                x = self.bn(x)
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dim, output_dims, activations, bn: bool, use_bias=True, drop=None):
        super(FC, self).__init__()
        assert type(output_dims) == type(activations) == list
        self.mods = nn.ModuleList()
        for output_dim, activation in zip(output_dims, activations):
            if drop is not None and drop != 'None':
                self.mods.append(nn.Dropout(drop))
            self.mods.append(conv2d(input_dim, output_dim, activation, bn, use_bias))
            input_dim = output_dim

    def forward(self, x):
        for mod in self.mods:
            x = mod(x)
        return x


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: [length, d_model] position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def positional_encoding_traffic(batch_size, time_step, node_number, d_model):
    """
    traffic data's shape is [batch_size, time_step, node_number, d_model].
    we only focus on time_step
    """
    pe = positional_encoding_1d(d_model, time_step)  # [time_step, d_model]
    pe = torch.stack([pe] * node_number, dim=1)  # [time_step, node_number, d_model]
    pe = torch.stack([pe] * batch_size, dim=0)  # [batch_size, time_step, node_number, d_model]
    return pe

import numpy as np

import torch
import torch.nn as nn
import random
from baselines.utils import calculate_random_walk_matrix


# according to DCRNN pytorch implementation
class DiffusionGCN(nn.Module):
    def __init__(self, supports, node_num, dim_in, dim_out, order, kernel='conv'):
        # order must be integer
        super(DiffusionGCN, self).__init__()
        self.node_num = node_num
        self.supports = supports
        self.supports_len = len(supports)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.order = order
        self.kernel = kernel
        self.weight = nn.Parameter(torch.FloatTensor(size=(dim_in * (order * self.supports_len + 1), dim_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(dim_out,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=0.)

    def forward(self, x):
        # shape of x is [B, N, D]
        batch_size = x.shape[0]
        # print(x.shape[1] , self.node_num , self.dim_in , x.shape[2])
        assert x.shape[1] == self.node_num and self.dim_in == x.shape[2]

        out = [x]
        x0 = x
        for support in self.supports:
            x1 = torch.einsum('ij, bjk -> bik', support, x0)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = 2 * torch.einsum('ij, bjk -> bik', support, x1) - x0
                out.append(x2)
                x1, x0 = x2, x1
        out = torch.cat(out, dim=-1)  # B, N, D, order
        out = out.reshape(batch_size * self.node_num, -1)  # B*N, D
        out = torch.matmul(out, self.weight)  # (batch_size * self._num_nodes, output_size)
        out = torch.add(out, self.biases)
        out = out.reshape(batch_size, self.node_num, self.dim_out)
        return out


class DCGRUCell(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order):
        super(DCGRUCell, self).__init__()
        self.kernel = 'mlp'  # kernel of GCN
        self.num_node = num_node
        self.hidden_dim = hidden_dim
        self.gate = DiffusionGCN(supports, num_node, input_dim + hidden_dim, 2 * hidden_dim, order, self.kernel)
        self.update = DiffusionGCN(supports, num_node, input_dim + hidden_dim, hidden_dim, order, self.kernel)

    def forward(self, x, state):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_node, self.hidden_dim)


class Encoder(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order, num_layers=1):
        super(Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DCGRUCell(supports, num_node, input_dim, hidden_dim, order))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(DCGRUCell(supports, num_node, hidden_dim, hidden_dim, order))

    def forward(self, x, init_state):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order, num_layers):
        super(Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_node = num_node
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decoder_cells = nn.ModuleList()
        self.decoder_cells.append(DCGRUCell(supports, num_node, input_dim, hidden_dim, order))
        for _ in range(1, num_layers):
            self.decoder_cells.append(DCGRUCell(supports, num_node, hidden_dim, hidden_dim, order))
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs, init_state, teacher_forcing_ratio):
        # shape of inputs: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        # if teacher_forcing=1, then teacher forcing in all steps
        # if teacher_forcing=0, then no teacher forcing
        seq_length = inputs.shape[1]
        outputs = []
        current_input = inputs[:, 0, :, :self.input_dim]
        for t in range(seq_length - 1):
            new_state = []
            for i in range(self.num_layers):
                state = init_state[i]
                state = self.decoder_cells[i](current_input, state)
                current_input = state
                new_state.append(state)
            init_state = torch.stack(new_state, dim=0)
            current_input = current_input.reshape(-1, self.hidden_dim)  ## [B, N, dim_out] to [B*N, D]
            current_input = self.projection(current_input)
            current_input = current_input.reshape(-1, self.num_node, self.input_dim)
            outputs.append(current_input)
            # in the val and test phase, teacher_forcing_ratio=0
            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            if self.training and teacher_force:
                current_input = inputs[:, t + 1, :, :self.input_dim]
        return torch.stack(outputs, dim=1)  # B, T, N, dim_in


class DCRNN(nn.Module):
    def __init__(self, adj_mx, num_nodes, input_dim, rnn_units, output_dim, max_diffusion_step, device, num_rnn_layers,
                 window, horizon):
        super(DCRNN, self).__init__()

        supports = [calculate_random_walk_matrix(adj_mx)]  # todo dual
        supports = [torch.tensor(i).to(device).float() for i in supports]
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.window = window
        self.horizon = horizon
        self.encoder = Encoder(supports, num_nodes, input_dim, rnn_units, max_diffusion_step, num_rnn_layers)
        self.decoder = Decoder(supports, num_nodes, output_dim, rnn_units, max_diffusion_step, num_rnn_layers)

        self.to(self.device)

    def forward(self, source, **kwargs):
        if 'real' in kwargs:
            # training
            assert self.training
            targets = kwargs.get('real')
            teacher_forcing_ratio = 0.5
        else:
            # validate
            assert not self.training
            targets = torch.zeros(source.shape[0], self.horizon, self.num_node, self.output_dim).to(self.device)
            teacher_forcing_ratio = -1

        # source: B, T_1, N, D
        # target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        _, encoder_hidden_state = self.encoder(source, init_state)
        GO_Symbol = torch.zeros(targets.shape[0], 1, self.num_node, self.input_dim).to(self.device)
        targets_len = self.horizon
        # targets = torch.cat([GO_Symbol, targets], dim=1)[:, 0:targets_len, ...]  # B, T, N, D to B, T, N, D
        targets = torch.cat([GO_Symbol, targets], dim=1)
        outputs = self.decoder(targets, encoder_hidden_state, teacher_forcing_ratio)
        return outputs  # B, T, N, D
        # return outputs[:, 1:, :, :]

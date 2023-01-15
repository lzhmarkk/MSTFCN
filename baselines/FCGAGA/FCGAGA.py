import torch
import torch.nn as nn
import torch.nn.functional as fn


class FcBlock(nn.Module):
    def __init__(self, block_layers, hidden_units, input_size, output_size):
        super(FcBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.block_layers = block_layers
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_units, bias=True),
            nn.ReLU()
        ))
        for i in range(block_layers - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_units, hidden_units, bias=True),
                nn.ReLU()
            ))
        self.forecast = nn.Linear(hidden_units, self.output_size, bias=True)
        self.backcast = nn.Linear(hidden_units, self.input_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # inputs: (B, N, D + N + N * T, C)
        inputs = inputs.permute(0, 1, 3, 2)  # (B, N, C, D + N + N * T)
        h = self.fc_layers[0](inputs)  # (B, N, C, D)
        for i in range(1, self.block_layers):
            h = self.fc_layers[i](h)  # (B, N, C, D)
        backcast = self.relu(inputs - self.backcast(h))
        return backcast.permute(0, 1, 3, 2), self.forecast(h).permute(0, 1, 3, 2)


class FcGagaLayer(nn.Module):
    def __init__(self, n_blocks, block_layers, hidden_units, node_id_dim, input_dim, output_dim, num_nodes,
                 window, horizon, epsilon):
        super(FcGagaLayer, self).__init__()
        self.num_nodes = num_nodes
        self.concat_dim = node_id_dim + window + self.num_nodes * window  # C + T + N * T
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.window = window
        self.horizon = horizon
        self.eps = 1e-8

        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(FcBlock(block_layers=block_layers,
                                       hidden_units=hidden_units,
                                       input_size=self.concat_dim,
                                       output_size=self.horizon))

        self.node_id_em = nn.Embedding(self.num_nodes,
                                       node_id_dim)
        nn.init.xavier_uniform_(self.node_id_em.weight)

        self.time_gate1 = nn.Sequential(
            nn.Linear(node_id_dim + window, hidden_units, bias=True),
            nn.ReLU()
        )
        self.time_gate2 = nn.Linear(hidden_units, horizon, bias=True)
        self.time_gate3 = nn.Linear(hidden_units, window, bias=True)

    def divide_no_nan(self, input, other):
        # assert not other.isnan().any()
        # assert not input.isnan().any()
        other = other + self.eps
        input = torch.divide(input, other)
        input = torch.where(input.isnan(), 0, input)
        input = torch.where(input.isinf(), 0, input)
        return input

    def forward(self, history_in, time_of_day_in):
        """
        :parameters
        history_in: (B, N, T, C)
        time_of_day_in: (B, N, T)
        """
        bs = history_in.shape[0]
        t = history_in
        # assert not torch.isinf(history_in).any()
        # assert not torch.isnan(history_in).any()

        node_embeddings = self.node_id_em.weight  # (N, D)
        node_id = node_embeddings.unsqueeze(0).repeat(bs, 1, 1)  # (B, N, D)

        time_gate = self.time_gate1(torch.concat([node_id, time_of_day_in], dim=-1))
        time_gate_forward = self.time_gate2(time_gate)  # (B, N, T)
        time_gate_backward = self.time_gate3(time_gate)  # (B, N, T)
        history_in = history_in / (1.0 + time_gate_backward.unsqueeze(-1))  # (B, N, T, C)

        # assert not torch.isnan(time_gate_forward).any()
        # assert not torch.isinf(time_gate_forward).any()
        # assert not torch.isnan(time_gate_backward).any()
        # assert not torch.isinf(time_gate_backward).any()
        # assert not torch.isnan(history_in).any()
        # assert not torch.isinf(history_in).any()

        node_embeddings_dp = torch.matmul(node_embeddings, node_embeddings.transpose(1, 0))
        node_embeddings_dp = torch.exp(self.epsilon * node_embeddings_dp)  # (N, N)
        # node_embeddings_dp = torch.where(node_embeddings_dp > 1e4, 1e4, node_embeddings_dp)
        node_embeddings_dp = node_embeddings_dp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, N, N, 1, 1)

        # assert not torch.isnan(node_embeddings_dp).any()
        # assert not torch.isinf(node_embeddings_dp).any()

        level = torch.max(history_in, dim=2, keepdim=True).values  # (B, N, 1, C)
        history = self.divide_no_nan(history_in, level)  # (B, N, T, C)

        # assert not torch.isnan(history).any()
        # assert not torch.isinf(history).any()
        # Add history of all other nodes
        shape = history_in.shape
        all_node_history = torch.tile(history_in.unsqueeze(1), [1, self.num_nodes, 1, 1, 1])  # (B, N, N, T, C)

        all_node_history = all_node_history * node_embeddings_dp  # (B, N, N, T, C)
        all_node_history = all_node_history.reshape(-1, self.num_nodes, self.num_nodes * shape[2],
                                                    shape[-1])  # (B, N, N * T, C)
        all_node_history = self.divide_no_nan(all_node_history - level, level)  # (B, N, N * T, C)

        # assert not torch.isnan(all_node_history).any()
        # assert not torch.isinf(all_node_history).any()

        all_node_history = fn.relu(all_node_history)  # (B, N, N * T, C)
        history = torch.concat([history, all_node_history], dim=2)
        # Add node ID
        # (B, N, N * T + T + D, C)
        history = torch.concat([history, node_id.unsqueeze(-1).expand(-1, -1, -1, shape[-1])], dim=2)

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.n_blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block

            # assert not torch.isnan(forecast_out).any()
            # assert not torch.isinf(forecast_out).any()
            # assert not torch.isnan(backcast).any()
            # assert not torch.isinf(backcast).any()

        # forecast_out = forecast_out[:, :, :self.horizon]# (B, N, T)
        forecast = forecast_out * level  # (B, N, T, C)

        forecast = (1.0 + time_gate_forward).unsqueeze(-1) * forecast
        # assert not torch.isnan(forecast).any()
        # assert not torch.isinf(forecast).any()
        # assert not torch.isnan(backcast).any()
        # assert not torch.isinf(backcast).any()

        return backcast, forecast


class FCGAGA(nn.Module):
    def __init__(self, device, n_stacks, n_blocks, block_layers, hidden_units, node_id_dim, input_dim, output_dim,
                 num_nodes, window, horizon, epsilon):
        super(FCGAGA, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.n_stack = n_stacks
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcgaga_layers = nn.ModuleList()
        for i in range(n_stacks):
            self.fcgaga_layers.append(
                FcGagaLayer(n_blocks=n_blocks, block_layers=block_layers, hidden_units=hidden_units,
                            node_id_dim=node_id_dim, num_nodes=num_nodes, input_dim=input_dim, output_dim=output_dim,
                            window=window, horizon=horizon, epsilon=epsilon))

        # self.start_conv = nn.Linear(input_dim, 1, bias=True)
        # self.end_conv = nn.Linear(1, output_dim, bias=True)

        self.to(self.device)

    def forward(self, input, **kwargs):
        x = input[..., :-2].permute(0, 2, 1, 3)  # (B, N, T, C)
        time = input[..., -2].permute(0, 2, 1)  # (B, N, T)

        # x = self.start_conv(x).squeeze(-1)
        backcast, forecast = self.fcgaga_layers[0](x, time)

        # assert not torch.isnan(backcast).any()
        # assert not torch.isnan(forecast).any()

        backcast, forecast_graph = self.fcgaga_layers[1](forecast, time)
        # assert not torch.isnan(backcast).any()
        # assert not torch.isnan(forecast_graph).any()
        forecast = forecast + forecast_graph

        backcast, forecast_graph = self.fcgaga_layers[2](forecast, time)
        # assert not torch.isnan(backcast).any()
        # assert not torch.isnan(forecast_graph).any()
        forecast = forecast + forecast_graph

        forecast = forecast / self.n_stack
        # forecast = torch.where(torch.isnan(forecast), torch.zeros_like(forecast), forecast)
        # forecast = self.end_conv(forecast.unsqueeze(-1))
        forecast = forecast.permute(0, 2, 1, 3)
        assert not torch.isnan(forecast).any()
        return forecast

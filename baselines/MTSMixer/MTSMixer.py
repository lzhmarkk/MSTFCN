import torch
import torch.nn as nn
import torch.nn.functional as fn


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        # self.dropouts = nn.ModuleList()
        self.individual = individual

    def forward(self, x):
        # x: [B, T, N]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, T, N]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, T, N]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class MTSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, n_layers, n_nodes, d_model, d_ff, fac_T, fac_C, sampling, norm, individual):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(seq_len, n_nodes, d_model, d_ff, fac_T, fac_C, sampling, norm) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(n_nodes) if norm else None
        self.projection = ChannelProjection(seq_len, pred_len, n_nodes, individual)
        # self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.refine = MLPBlock(configs.pred_len, configs.d_model) if configs.refine else None

    def forward(self, x, **kwargs):
        # x = self.rev(x, 'norm') if self.rev else x

        B, C = x.shape[0], x.shape[3]
        x = x.permute(0, 3, 1, 2)  # (B,C,T,N)
        x = x.reshape(B * C, *x.shape[-2:])  # (B*C,T,N)

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        # x = self.refine(x.transpose(1, 2)).transpose(1, 2) if self.refine else x
        # x = self.rev(x, 'denorm') if self.rev else x

        x = x.reshape(B, C, *x.shape[-2:])
        x = x.permute(0, 2, 3, 1)

        return x

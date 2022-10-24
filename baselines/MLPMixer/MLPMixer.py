import torch
import torch.nn as nn


class MixerLayer(nn.Module):
    def __init__(self, channel, dropout):
        super(MixerLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(channel)
        self.fc1 = nn.Linear(channel, channel, bias=True)
        self.fc2 = nn.Linear(channel, channel, bias=True)
        self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (..., channel)
        h = self.layer_norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = x + h
        return h


class FFTLayer(nn.Module):
    def __init__(self, seq_len, hidden_dim, fft_dropout):
        super(FFTLayer, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.fft_dropout = fft_dropout

        self.complex_weight = nn.Parameter(torch.randn(1, seq_len // 2 + 1, hidden_dim, 2, dtype=torch.float32) * 0.02,
                                           requires_grad=True)
        self.out_dropout = nn.Dropout(fft_dropout)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)

    def forward(self, x):
        bs, n_nodes, seq_len, hidden = x.shape
        h = torch.fft.rfft(x, dim=2, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        h = h * weight
        h = torch.fft.irfft(h, n=seq_len, dim=2, norm='ortho')
        h = self.out_dropout(h)
        h = self.LayerNorm(h + x)
        return h


class MLPMixer(nn.Module):
    def __init__(self, device, input_dim, output_dim, window, horizon, hidden_dim, dropout, num_nodes, use_fft, fft_dropout):
        super(MLPMixer, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window = window
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_fft = use_fft

        self.num_nodes = num_nodes

        if use_fft:
            self.time_fft = FFTLayer(window, hidden_dim, dropout)
        else:
            self.time_mixer = MixerLayer(window, dropout)
        self.node_mixer = MixerLayer(num_nodes, dropout)
        self.channel_mixer = MixerLayer(hidden_dim, dropout)

        self.embedding_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x, **kwargs):
        x = x[..., :self.input_dim]
        # (bs, window, num_nodes, input_dim)
        h = self.embedding_layer(x)
        h = self.channel_mixer(h)  # (bs, window, num_nodes, hidden_dim)
        if self.use_fft:
            h = h.permute(0, 2, 1, 3)  # (bs, num_nodes, window, hidden_dim)
            h = self.time_fft(h)
            h = h.permute(0, 3, 2, 1)  # (bs, hidden_dim, window, num_nodes)
        else:
            h = h.permute(0, 3, 2, 1)  # (bs, hidden_dim, num_nodes, window)
            h = self.time_mixer(h)
            h = h.permute(0, 1, 3, 2)  # (bs, hidden_dim, window, num_nodes)
        h = self.node_mixer(h)
        h = h.permute(0, 2, 3, 1)  # (bs, window, num_nodes, hidden_dim)
        h = self.output_layer(h)
        return h

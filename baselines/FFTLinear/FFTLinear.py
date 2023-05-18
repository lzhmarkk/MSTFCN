import torch
import torch.nn as nn
import torch.nn.functional as fn


class FFTLinear(nn.Module):
    def __init__(self, n_dim, n_nodes, window, horizon, n_hidden, input_dim, output_dim, add_time):
        super().__init__()

        self.n_dim = input_dim
        self.n_nodes = n_nodes
        self.window = window
        self.horizon = horizon
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.ModuleList()
        for _ in range(input_dim):
            self.linear.append(nn.Linear(window, horizon))

        # todo time semantic weight and bias
        # self.weight = nn.Parameter(torch.randn([self.n_dim, window // 2 + 1, window // 2 + 1], dtype=torch.cfloat))
        # self.bias = nn.Parameter(torch.randn([1, self.n_dim, 1, window // 2 + 1], dtype=torch.cfloat))

        assert not add_time

    def forward(self, input, **kwargs):
        x = input.transpose(3, 1)  # (B, C, N, T)

        y = [self.linear[_](x[:, _, :, :]) for _ in range(2)]
        y = torch.stack(y, dim=1)

        # h = torch.fft.rfft(x, dim=-1, norm='ortho')
        # assert torch.is_complex(h)
        #
        # h = torch.einsum("bcnt,ctf->bcnf", h, self.weight)
        # h = h + self.bias
        #
        # y = torch.fft.irfft(h, dim=-1, n=self.horizon, norm='ortho')
        y = y.transpose(3, 1)
        return y

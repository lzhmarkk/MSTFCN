import torch
import torch.nn as nn


class MixedFusion(nn.Module):
    def __init__(self, in_dim: int):
        super(MixedFusion, self).__init__()
        self.in_dim = in_dim
        # self.lin_A = nn.Linear(in_dim ** 2, in_dim ** 2)
        # self.lin_P = nn.Linear(in_dim ** 2, in_dim ** 2)
        self.wa = nn.Parameter(torch.randn(in_dim, in_dim), requires_grad=True)
        self.wp = nn.Parameter(torch.randn(in_dim, in_dim), requires_grad=True)
        self.b = nn.Parameter(torch.randn(in_dim, in_dim), requires_grad=True)

    def forward(self, A: torch.Tensor, P: torch.Tensor):
        assert len(A.shape) == len(P.shape) == 2
        # _A, _P = A.reshape(self.in_dim * self.in_dim), P.reshape(self.in_dim * self.in_dim)
        # a_A = self.lin_A(_A)
        # a_P = self.lin_P(_P)
        # a = torch.sigmoid(torch.add(a_A, a_P)).reshape(self.in_dim, self.in_dim)
        a = torch.sigmoid(self.wa * A + self.wp * P + self.b)
        G = torch.add(torch.mul(a, A), torch.mul(1 - a, P))

        return G


class MGP_Gen(nn.Module):
    def __init__(self, num_nodes: int, num_categories: int, hidden_dim: int, alpha: int = 3):
        super(MGP_Gen, self).__init__()
        self.alpha = alpha
        self.params_S = self.init_params(num_categories, hidden_dim)
        self.aggreg_S = MixedFusion(num_nodes)
        self.params_C = self.init_params(num_nodes, hidden_dim)
        self.aggreg_C = MixedFusion(num_categories)

    def init_params(self, in_dim: int, hidden_dim: int):
        params = nn.ParameterDict()
        params['Wu'] = nn.Parameter(torch.randn(in_dim, hidden_dim), requires_grad=True)
        params['Wv'] = nn.Parameter(torch.randn(in_dim, hidden_dim), requires_grad=True)
        for param in params.values():
            nn.init.xavier_normal_(param)
        return params

    def forward(self, X_seq: torch.Tensor, As: torch.Tensor, Ac: torch.Tensor):
        # branch S
        Us = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_S['Wu']))
        Vs = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_S['Wv']))
        Ps = torch.einsum('btnh,btmh->nm', Us, Vs) - torch.einsum('btmh,btnh->mn', Vs, Us)
        Ps = torch.softmax(torch.relu(Ps), dim=-1)
        Gs = self.aggreg_S(As, Ps)

        # branch C
        X_seq = X_seq.transpose(2, 3)
        Uc = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_C['Wu']))
        Vc = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_C['Wv']))
        Pc = torch.einsum('btnh,btmh->nm', Uc, Vc) - torch.einsum('btmh,btnh->mn', Vc, Uc)
        Pc = torch.softmax(torch.relu(Pc), dim=-1)
        Gc = self.aggreg_C(Ac, Pc)

        return Gs, Gc

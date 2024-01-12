import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FC, positional_encoding_traffic


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, K, d, bn):
        super(CrossModalAttention, self).__init__()
        self.d = d
        self.K = K
        self.D = K * d
        self.w_qkv = FC(input_dim * 2, [self.D * 2], [F.gelu], bn)
        self.ffn = FC(self.D, [self.D, self.D], [F.gelu, None], bn)

    def forward(self, input_list: list, STE):
        x = torch.cat(input_list, dim=1)
        x_res = x
        x = torch.cat([x, torch.cat([STE] * len(input_list), dim=1)], dim=-1)

        # [batch_size, 3 * num_step, N, 2 * K * d]
        query_key_value = self.w_qkv(x)
        # [K * batch_size, 3 * num_step, N, 2 * d]
        query_key_value = torch.cat(torch.chunk(query_key_value, self.K, -1), 0)

        # [K * batch_size, 3 * num_step, N, d]
        query_key = query_key_value[:, :, :, :self.d]
        value = query_key_value[:, :, :, self.d:]

        # query: [K * batch_size, N, 3 * num_step, d]
        # key:   [K * batch_size, N, d, 3 * num_step]
        # value: [K * batch_size, N, 3 * num_step, d]
        value = value.permute(0, 2, 1, 3)

        # [K * batch_size, N, 3 * num_step, 3 * num_step]
        attention = torch.matmul(query_key.permute(0, 2, 1, 3), query_key.permute(0, 2, 3, 1))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, -1)

        # [batch_size, 3 * num_step, N, D]
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3)
        x = torch.cat(torch.chunk(x, self.K, 0), -1)
        x += x_res
        # Position-wise Feed-Forward Network
        x = self.ffn(x) + x
        return torch.split(x, input_list[0].shape[1], dim=1)


class PositionalEncoding(nn.Module):
    '''
    Positional Encoding
    '''

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, shape, device):
        '''
        shape: Size(batch_size, P, N, D)
        device: tensor's device type
        return: [batch_size, P, N, D]
        '''
        batch_size, P, N, D = shape
        return positional_encoding_traffic(batch_size, P, N, D).to(device)


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, T + 7 + 7 + 2]
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''

    def __init__(self, T, D, bn):
        super(STEmbedding, self).__init__()
        self.fc_s = FC(D, [D, D], [F.gelu, None], bn)
        self.fc_t = FC(T + 7 + 7 + 2, [D, D], [F.gelu, None], bn)
        self.T = T

    def forward(self, SE, TE: torch.LongTensor):
        # spatial embedding
        # SE [N, D] -> [1, 1, N, D] -> [1, 1, N, D]
        SE = torch.unsqueeze(torch.unsqueeze(SE, 0), 0)
        SE = self.fc_s(SE)

        # temporal embedding
        # dow: [batch, P+Q, 7]
        # tod: [batch, P+Q, T]
        # TE: [batch, P+Q, T+7] -> [batch, P+Q, 1, T+7] -> [batch, P+Q, 1, D]
        dayofweek = F.one_hot(TE[..., 0], 7).float()
        timeofday = F.one_hot(TE[..., 1], self.T).float()
        # Time Series 6&8
        Peri_72 = F.one_hot(TE[..., 2], 4).float()
        Peri_96 = F.one_hot(TE[..., 3], 3).float()
        Peri_144 = F.one_hot(TE[..., 4], 2).float()

        TE = torch.cat([dayofweek, timeofday, Peri_72, Peri_96, Peri_144], dim=-1)
        TE = torch.unsqueeze(TE, dim=2)
        TE = self.fc_t(TE)

        return SE + TE


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, K, d, bn):
        """
        input_dim: input's dimension
        K: number of attention heads
        d: dimension of each attention outputs
        """
        super(SpatialAttention, self).__init__()
        self.d = d
        self.K = K
        self.D = K * d
        self.w_qkv = FC(input_dim * 2, [self.D * 2], [F.gelu], bn)
        self.ffn = FC(self.D, [self.D, self.D], [F.gelu, None], bn)

    def forward(self, x, STE):
        """
        X: [batch_size, num_step, N, D]
        STE: [batch_size, num_step, N, D]
        return: [batch_size, num_step, N, D]
        """
        x_res = x
        x = torch.cat([x, STE], dim=-1)

        # [batch_size, num_step, N, 2 * K * d]
        query_key_value = self.w_qkv(x)
        # [K * batch_size, num_step, N, 2 * d]
        query_key_value = torch.cat(torch.chunk(query_key_value, self.K, -1), 0)

        # [K * batch_size, num_step, N, d]
        query_key = query_key_value[:, :, :, :self.d]
        value = query_key_value[:, :, :, self.d:]

        # [K * batch_size, num_step, N, N]
        attention = torch.matmul(query_key, query_key.permute(0, 1, 3, 2))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, -1)

        # [batch_size, num_step, N, D]
        x = torch.matmul(attention, value)
        x = torch.cat(torch.chunk(x, self.K, 0), -1)
        x = x + x_res
        # Position-wise Feed-Forward Network
        x = self.ffn(x) + x
        return x


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, K, d, bn):
        """
        input_dim: input's dimension
        K: number of attention heads
        d: dimension of each attention outputs
        """
        super(TemporalAttention, self).__init__()
        self.d = d
        self.K = K
        self.D = K * d
        self.w_qkv = FC(input_dim * 2, [self.D * 2], [F.gelu], bn)
        self.ffn = FC(self.D, [self.D, self.D], [F.gelu, None], bn)

    def forward(self, x, STE):
        """
        X: [batch_size, num_step, N, D]
        STE: [batch_size, num_step, N, D]
        return: [batch_size, num_step, N, D]
        """
        x_res = x
        x = torch.cat([x, STE], dim=-1)

        # [batch_size, num_step, N, 2 * K * d]
        query_key_value = self.w_qkv(x)
        # [K * batch_size, num_step, N, 2 * d]
        query_key_value = torch.cat(torch.chunk(query_key_value, self.K, -1), 0)

        # [K * batch_size, num_step, N, d]
        query_key = query_key_value[:, :, :, :self.d]
        value = query_key_value[:, :, :, self.d:]

        # query: [K * batch_size, N, num_step, d]
        # key:   [K * batch_size, N, d, num_step]
        # value: [K * batch_size, N, num_step, d]
        # query_key = query_key.permute(0,2,1,3)
        # key = key.permute(0,2,3,1)
        value = value.permute(0, 2, 1, 3)

        # [K * batch_size, N, num_step, num_step]
        attention = torch.matmul(query_key.permute(0, 2, 1, 3), query_key.permute(0, 2, 3, 1))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, -1)

        # [batch_size, num_step, N, D]
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3)
        x = torch.cat(torch.chunk(x, self.K, 0), -1)
        x = x + x_res
        # Position-wise Feed-Forward Network
        x = self.ffn(x) + x
        return x


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, N, input_dim, K, d, bn):
        super(CrossModalAttentionBlock, self).__init__()
        self.mods = nn.ModuleList()
        for _ in range(N):
            self.mods.append(CrossModalAttention(input_dim, K, d, bn))

    def forward(self, input_list: list, STE_P):
        input_list_ = input_list
        for mod in self.mods:
            input_list = mod(input_list, STE_P)
        return [input_list_[i] + input_list[i] for i in range(len(input_list))]


class STAttentionBlock(nn.Module):
    def __init__(self, input_dim, K, d, bn):
        super(STAttentionBlock, self).__init__()
        self.t_att = TemporalAttention(input_dim, K, d, bn)
        self.s_att = SpatialAttention(input_dim, K, d, bn)

    def forward(self, x, STE):
        HT = self.t_att(x, STE)
        HS = self.s_att(HT, STE)
        return x + HS


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, input_list: list):
        x = input_list[0]
        for i in range(1, len(input_list)):
            x = torch.max(x, input_list[i])
        return x


class ST_MAN(nn.Module):
    def __init__(self, input_dim, P, Q, T=24 * 60 // 30, N=2, L=1, K=4, d=8, drop_rate=0.0, bn=True):
        super(ST_MAN, self).__init__()
        D = K * d
        self.P = P
        self.Q = Q

        self.input_embedding_1 = FC(input_dim, [D, D], [F.gelu, None], bn)
        self.input_embedding_2 = FC(input_dim, [D, D], [F.gelu, None], bn)
        self.input_embedding_3 = FC(input_dim, [D, D], [F.gelu, None], bn)

        self.positional_encoding = PositionalEncoding()
        self.ste_embedding = STEmbedding(T, D, bn)

        self.encoder = CrossModalAttentionBlock(N, D, K, d, bn)

        self.max_pooling = MaxPooling()

        self.decoder = nn.ModuleList()
        for _ in range(L):
            self.decoder.append(STAttentionBlock(D, K, d, bn))

        self.output_fc = FC(D, [D, 1], [F.gelu, None], bn, use_bias=True, drop=drop_rate)

    def forward(self, x, TE, SE):
        raise ValueError()
        # now is [...,..., 3]
        # x = torch.unsqueeze(x, -1)
        x_1 = self.input_embedding_1(x[:, :, :, 0].unsqueeze(-1))
        x_2 = self.input_embedding_2(x[:, :, :, 1].unsqueeze(-1))
        x_3 = self.input_embedding_3(x[:, :, :, 2].unsqueeze(-1))
        # after input embedding, we add positional encoding information to x
        pe = self.positional_encoding(x_1.shape, x_1.device)
        x_1 = x_1 + pe
        x_2 = x_2 + pe
        x_3 = x_3 + pe
        pe = None

        STE = self.ste_embedding(SE, TE)

        x_1, x_2, x_3 = self.encoder([x_1, x_2, x_3], STE[:, :self.P])

        x = self.max_pooling([x_1, x_2, x_3])

        for mod in self.decoder:
            x = mod(x, STE[:, self.P:])

        x = self.output_fc(x)
        return x  # return [...,..., 1]

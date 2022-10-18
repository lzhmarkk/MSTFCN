import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, device, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.device = device
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        x = x.to(self.device)
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, device, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, device=device)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, device=device)  # input_dims = time step per day + days per week=12+7=19
        self.device = device

    def forward(self, SE, TE, T=288):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(self.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(self.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 1][i].to(torch.int64) % 7, 7).to(self.device)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 0][j].to(torch.int64) % T, T).to(self.device)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        _, _, num_nodes, _ = SE.shape
        TE = TE.expand(TE.shape[0], TE.shape[1], num_nodes, TE.shape[3])
        return TE
        # return SE + TE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, device):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, device=device)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, device, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.device = device
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, device=device)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step).to(self.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay, device):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False, device=device)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True, device=device)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay, device=device)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, device, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay, device)
        self.temporalAttention = temporalAttention(K, d, device, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay, device)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        # print('HS.shape: ', HS.shape)
        HT = self.temporalAttention(X, STE)
        # print('HT.shape: ', HT.shape)
        H = self.gatedFusion(HS, HT)
        # print('H.shape: ', H.shape)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, device):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, device=device)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, device=device)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    # todo tune GMAN hyper parameter to improve performance, 20221016 lzhmark
    def __init__(self, SE, device, L, K, d, bn_decay, window, input_dim, output_dim):
        super().__init__()
        self.SE = SE
        self.device = device

        self.L = L
        self.K = K
        self.d = d
        self.D = self.K * self.d
        self.bn_decay = bn_decay
        self.num_his = window
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.STEmbedding = STEmbedding(self.D, self.device, self.bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(self.K, self.d, self.device, self.bn_decay) for _ in range(self.L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(self.K, self.d, self.device, self.bn_decay) for _ in range(self.L)])
        self.transformAttention = transformAttention(self.K, self.d, self.bn_decay, self.device)
        self.FC_1 = FC(input_dims=[1, self.D], units=[self.D, self.D], activations=[F.relu, None],
                       bn_decay=self.bn_decay, device=self.device)
        self.FC_2 = FC(input_dims=[self.D, self.D], units=[self.D, 1], activations=[F.relu, None],
                       bn_decay=self.bn_decay, device=self.device)

        self.to(self.device)

    def forward(self, input, **kwargs):
        # input
        # print('X.shape: ', X.shape) #[batch_size, seq_length, nodes_num, input_dim + time]
        X = input[..., :self.input_dim]  # [batch_size, seq_length, nodes_num, input_dim]
        X = X.reshape(X.shape[0], X.shape[1], -1).unsqueeze(dim=-1)
        his_time = input[:, :, 0, self.input_dim:]
        pred_time = kwargs['pred_time'][:, :, 0, :]
        TE = torch.cat([his_time, pred_time], 1)  # [batch_size, seq_length, time=2]
        # print('X.shape: ', X.shape)
        X = self.FC_1(X)  # (bs, window, num_nodes*input_dim, self.D)
        # print('X.shape: ', X.shape)
        # STE
        # print('TE.shape: ', TE.shape) #[batch_size, his+pre, 2]
        STE = self.STEmbedding(self.SE, TE)
        # print('STE.shape: ', STE.shape)
        STE_his = STE[:, :self.num_his]  # (bs, window, num_nodes, self.D)
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # print('X.shape: ', X.shape)
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # output
        # print('X.shape: ', X.shape)
        X = self.FC_2(X)
        # print('X.shape: ', X.shape)
        del STE, STE_his, STE_pred
        # exit()
        X = X.reshape(X.shape[0], X.shape[1], input.shape[2], self.output_dim)
        return X  # [batch_size, seq_length, num_nodes, output_dim]

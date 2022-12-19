import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    """

    def __init__(self, adj_mx, device, num_nodes, input_dim, output_dim, window, horizon,
                 spatial_channels, hidden_channel):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()

        self.adj_mx = torch.from_numpy(adj_mx).to(device).float()
        self.device = device

        num_nodes = num_nodes
        num_features = input_dim
        num_timesteps_input = window
        num_timesteps_output = horizon

        self.block1 = STGCNBlock(in_channels=num_features, out_channels=hidden_channel,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=hidden_channel, out_channels=hidden_channel,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=hidden_channel, out_channels=hidden_channel)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * hidden_channel,
                               num_timesteps_output * output_dim)
        self.num_nodes = num_nodes
        self.input_dim = num_features
        self.output_dim = output_dim
        self.seq_len = num_timesteps_input
        self.horizon = num_timesteps_output

        self.to(device)

    def forward(self, X, **kwargs):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = X.permute(0, 2, 1, 3).contiguous()  # (bs, n_nodes, window, input_dim).
        # X = X.view(self.seq_len, -1, self.num_nodes, self.input_dim).permute(1, 2, 0, 3)
        out1 = self.block1(X, self.adj_mx)
        out2 = self.block2(out1, self.adj_mx)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))  # (bs, num_nodes, horizon * output_dim)
        out5 = out4.reshape(out4.shape[0], self.num_nodes, self.horizon, self.output_dim)
        return out5.permute(0, 2, 1, 3)

        # :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
        #          y: shape (horizon, batch_size, num_sensor * output_dim)

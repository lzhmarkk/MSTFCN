import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    对称归一化的拉普拉斯
    Args:
        adj: adj matrix
    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    L = D^-1 * A
    随机游走拉普拉斯
    Args:
        adj_mx: adj matrix
    Returns:
        np.ndarray: L
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    计算近似后的拉普莱斯矩阵~L
    Args:
        adj_mx:
        lambda_max:
        undirected:
    Returns:
        ~L = 2 * L / lambda_max - I
    """
    adj_mx = sp.coo_matrix(adj_mx)
    if undirected:
        bigger = adj_mx > adj_mx.T
        smaller = adj_mx < adj_mx.T
        notequall = adj_mx != adj_mx.T
        adj_mx = adj_mx - adj_mx.multiply(notequall) + adj_mx.multiply(bigger) + adj_mx.T.multiply(smaller)
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).tocoo()

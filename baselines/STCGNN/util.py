import numpy as np
from data.util import StandardScaler


def gen_Ac(x, delta=0.4):
    print("Generating A for categories...")
    T, N, C = x.shape
    x = x.astype(float)
    sim_matrix = np.zeros([C, C], dtype=float)

    for i in range(C):
        x[:, :, i] = StandardScaler(x[:, :, i]).transform(x[:, :, i])

    for i in range(C):
        for j in range(C):
            ix = x[:, :,i]
            iy = x[:, :,j]

            a = np.multiply(ix, iy).sum()
            b = np.sqrt(np.power(ix, 2).sum() * np.power(iy, 2).sum())
            sim_matrix[i][j] = a / b

    sim_matrix[sim_matrix < delta] = 0
    return sim_matrix

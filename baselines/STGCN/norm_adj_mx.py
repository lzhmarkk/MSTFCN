import numpy as np


def preprocess_adj(A):
    A[A > 0] = 1
    return A


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    # A = preprocess_adj(A)

    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1))).astype(float)
    return A_wave

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from data.load_dataset import load_dataset

dataset = 'chicago-bike'
n_days = 7
top = 10

if __name__ == '__main__':
    dataloader = load_dataset(dataset, 64, 12, 12, 2, 2, add_time=False)
    scaler = dataloader['scaler']
    N = dataloader['x_test'].shape[-2]
    y = dataloader['y_test'][0::12].reshape(-1, N, 2)

    idx = (y > 1).sum(0).sum(1)
    idx = np.argsort(idx)[-top:]

    truth = y[:n_days * 2 * 24]  # (T, N, C)
    # truth = scaler.inverse_transform(truth)

    fig, axs = plt.subplots(len(idx), figsize=(20, len(idx) * 4))
    for _, i in enumerate(idx):
        h = torch.from_numpy(truth[:, i, 0])
        f = torch.fft.rfft(h, dim=0, norm="ortho")  # (F,) complex
        f_real = f.real[1:]
        f_imag = f.imag[1:]

        ax = axs[_]
        ax.plot(range(len(h)), h.numpy())
        ax.plot(range(len(f_real)), f_real.numpy())
        ax.plot(range(len(f_imag)), f_imag.numpy())

    fig.tight_layout()
    plt.savefig(f"./fourier.png")
    # plt.legend()
    # plt.show()

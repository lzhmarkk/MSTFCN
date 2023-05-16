# if we remove 30% noises, the reconstruction v.s. window
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from data.load_dataset import load_dataset
from evaluate import masked_mae_np, masked_rmse_np

dataset = 'chicago-bike'
input_dim = 2
output_dim = 2
batch_size = 64
n_days = 14
top = 20
window = 96
drop = 0.3
length = n_days * 2 * 24
# n_filter = int(.5*length)
loss_fn = masked_mae_np

if __name__ == '__main__':
    dataloader = load_dataset(dataset, batch_size, 12, 12, input_dim, output_dim, add_time=False)
    scaler = dataloader['scaler']

    N = dataloader['x_test'].shape[-2]
    x = dataloader['x_test'][12::12].reshape(-1, N, input_dim)[:length]
    y = dataloader['y_test'][0::12].reshape(-1, N, output_dim)[:length]

    idx = (y > 1).sum(0).sum(1)
    idx = np.argsort(idx)[-top:]

    idx = 242
    x = torch.from_numpy(x[:, idx])  # (T, C)
    truth = y[:len(x), idx]  # (T, C)
    truth = scaler.inverse_transform(truth)

    losses_drop = []
    for drop in np.arange(0, 10) / 10:
        losses = []
        for window in range(8, 97):
            pred = []
            for i in range(math.ceil(x.shape[0] / window)):
                H = x[i * window:(i + 1) * window]
                F = torch.fft.rfft(H, dim=0, norm="ortho")  # (F, C)
                if drop > 0:
                    F = F[:-int(len(F) * drop)]
                _pred = torch.fft.irfft(F, dim=0, n=len(H), norm="ortho")
                pred.append(_pred)
            pred = torch.cat(pred, dim=0).numpy()
            pred = scaler.inverse_transform(pred)

            assert pred.shape == truth.shape
            loss = loss_fn(pred, truth, 0.)
            # loss = np.log(loss).item()
            losses.append(loss)
        losses_drop.append(losses)

    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.suptitle(f"window v.s. loss")
    for i, losses in enumerate(losses_drop):
        ax.plot(range(8, 97), losses, label=i)

    plt.legend()
    plt.show()

    # fig, axs = plt.subplots(2, figsize=(20, 10))
    # fig.suptitle(f"{window}")
    # for c in range(2):
    #     ax = axs[c]
    #     x = range(pred.shape[0])
    #     ax.plot(x, pred[..., c], label='pred')
    #     ax.plot(x, truth[..., c], label='truth')
    #
    # plt.legend()
    # plt.show()

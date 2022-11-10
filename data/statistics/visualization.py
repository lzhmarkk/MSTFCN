import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data.util import StandardScaler

dataset = 'chicago-taxi'
display_time_lmt = [0, 2 * 24 * 7 * 4]
selected_variables = [2, 30, 53, 70, 193]
selected_dim = 0
norm = False
drop_zero = False

if __name__ == '__main__':
    h5_path = os.path.join('../h5data', dataset + '.h5')
    df = h5py.File(h5_path, 'r')
    # rawdata = np.array(df['taxi_pick']).astype(float)
    rawdata = np.array(df['raw_data']).astype(float)

    # rawdata: (total_timestamps, num_nodes, input_dim)
    print("rawdata shape: ", rawdata.shape)
    # data = rawdata[...]
    data = rawdata[..., selected_dim]

    print(f"density {np.count_nonzero(data) / data.size}")

    # sort = []
    # for i in range(data.shape[1]):
    #     y = data[:, i]
    #     sort.append((i, y.std()))
    # sort = sorted(sort, key=lambda e: e[1])
    # print([e[1] for e in sort[:10]])
    # selected_variables = [e[0] for e in sort[:10]]

    if norm:
        scaler = StandardScaler(mean=data.mean(), std=data.std())
        data = scaler.transform(data)

    # draw
    for i in selected_variables:
        y = data[:, i]
        if drop_zero:
            y = y[y != 0]
        y = y[display_time_lmt[0]: display_time_lmt[1]]
        x = np.arange(y.shape[0])
        fig, ax = plt.subplots(figsize=(40, 10))
        plt.xticks([2 * 24 * day + 24 for day in range(7 * 4)], [f"day {day}" for day in range(7 * 4)])
        plt.plot(x, y)

        plt.savefig(f"../../saves/plot/{dataset}-{display_time_lmt[0]}-{display_time_lmt[1]}-{i}.png")
        plt.show()

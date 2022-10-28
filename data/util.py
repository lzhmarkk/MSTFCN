import os
import h5py
import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_graph_seq2seq_io_data(
        df, time, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape[0], df.shape[1]
    data_list = [df]

    if add_time_in_day:
        time = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S')
        time_ind = (time.values - time.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.tile(time.dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    print(data.shape)

    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(h5_name, window, horizon, add_time_in_day=False, add_day_in_week=False):
    # 30min
    h5_path = os.path.join('./data/h5data', h5_name + '.h5')
    df = h5py.File(h5_path, 'r')
    rawdata = np.array(df['raw_data']).astype(float)

    if 'time' in df:
        time = np.array(df['time']).astype(np.datetime64)
    else:
        time = None

    x_offsets = np.sort(
        np.concatenate((np.arange(-(window - 1), 1, 1),))
    )

    y_offsets = np.sort(np.arange(1, horizon + 1, 1))

    x, y = generate_graph_seq2seq_io_data(
        rawdata,
        time,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.7)
    num_val = round(num_samples * 0.15)
    num_test = num_samples - num_train - num_val

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    """
    save_folder =  os.path.join(h5_name)
    os.mkdir(save_folder)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)       
        np.savez_compressed(
            os.path.join(save_folder, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    """
    return [x_train, y_train], [x_val, y_val], [x_test, y_test]

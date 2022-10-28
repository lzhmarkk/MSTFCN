import math
import numpy as np
from data.util import StandardScaler, generate_train_val_test


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = math.ceil(self.size / self.batch_size)
        self.xs = xs
        self.ys = ys
        self.current_ind = 0

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __iter__(self):
        self.current_ind = 0
        return self

    def __next__(self):
        if self.current_ind < self.num_batch:
            start_ind = self.batch_size * self.current_ind
            end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
            x_i = self.xs[start_ind: end_ind, ...]
            y_i = self.ys[start_ind: end_ind, ...]
            self.current_ind += 1
            return x_i, y_i
        else:
            raise StopIteration()

    def __len__(self):
        return self.num_batch


def load_dataset(dataset, batch_size, window, horizon, input_dim, output_dim, valid_batch_size=None, test_batch_size=None, add_time=False):
    [x_train, y_train], [x_val, y_val], [x_test, y_test] = generate_train_val_test(dataset, window, horizon, add_time, add_time)

    scaler = StandardScaler(mean=x_train[..., :input_dim].mean(), std=x_train[..., :input_dim].std())

    x_train[..., : input_dim] = scaler.transform(x_train[..., : input_dim])
    y_train[..., : output_dim] = scaler.transform(y_train[..., : output_dim])
    x_val[..., : input_dim] = scaler.transform(x_val[..., : input_dim])
    y_val[..., : output_dim] = scaler.transform(y_val[..., : output_dim])
    x_test[..., : input_dim] = scaler.transform(x_test[..., : input_dim])
    y_test[..., : output_dim] = scaler.transform(y_test[..., : output_dim])

    valid_batch_size = valid_batch_size if valid_batch_size is not None else batch_size
    test_batch_size = test_batch_size if test_batch_size is not None else batch_size

    data = {}
    data['train_loader'] = DataLoaderM(x_train, y_train, batch_size)
    data['val_loader'] = DataLoaderM(x_val, y_val, valid_batch_size)
    data['test_loader'] = DataLoaderM(x_test, y_test, test_batch_size)
    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['scaler'] = scaler
    return data

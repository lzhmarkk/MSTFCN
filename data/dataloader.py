import numpy as np
import torch
import pandas as pd
from torch.autograd import Variable
from data.util import normal_std, StandardScaler, generate_train_val_test


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, dataset, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        filename = r'data/h5data/' + dataset + '.h5'
        self.rawdat = (pd.read_hdf(filename)).to_numpy()
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
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
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

    def get_one(self, index):
        start_ind = self.batch_size * index
        end_ind = min(self.size, self.batch_size * (index + 1))
        x_i = self.xs[start_ind: end_ind, ...]
        y_i = self.ys[start_ind: end_ind, ...]
        return (x_i, y_i)


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

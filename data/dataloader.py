import math
import numpy as np


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

import os
import numpy as np
from utils import env_paths
from parmesan.datasets import load_svhn
from data_helper import create_semi_supervised, pad_targets, cut_off_dataset


def _download(extra=False, normalize=True):
    """
    Download the SVHN dataset if it is not present.
    :return: The train, test and validation set.
    """

    def norm(x):
        x = x.reshape((-1, 3, 32 * 32))
        std = x.std(axis=(-1, 0))
        x[:, 0] /= std[0]
        x[:, 1] /= std[1]
        x[:, 2] /= std[2]
        x = x.reshape((-1, 3 * 32 * 32))
        return x

    train_x, train_t, test_x, test_t = load_svhn(os.path.join(env_paths.get_data_path("svhn"), ""),
                                                 normalize=False,
                                                 dequantify=True,
                                                 extra=extra)

    if normalize:
        train_x = norm(train_x)
        test_x = norm(test_x)

    train_t = np.array(train_t - 1, dtype='float32').reshape(-1)
    test_t = np.array(test_t - 1, dtype='float32').reshape(-1)

    # Dummy validation set. NOTE: still in training set.
    idx = np.random.randint(0, train_x.shape[0] - 1, 5000)
    valid_x = train_x[idx, :]
    valid_t = train_t[idx]

    return (train_x, train_t), (test_x, test_t), (valid_x, valid_t)


def _gen_conv(xy):
    x, y = xy
    x = x.reshape((-1, 3, 32, 32))
    return x, y


def load_supervised(conv=False, extra=False, normalize=True):
    """
    Load the SVHN dataset.
    :param conv: Boolean whether the images should be vectorized or not.
    :param extra: Include the extra set or not.
    :param normalize: Boolean normalize the data set.
    :return: The train, test and validation sets.
    """
    train_set, test_set, valid_set = _download(extra, normalize)

    test_set = pad_targets(test_set)
    train_set = pad_targets(train_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    if conv:
        test_set = _gen_conv(test_set)
        train_set = _gen_conv(train_set)
        if valid_set is not None:
            valid_set = _gen_conv(valid_set)
    return train_set, test_set, valid_set


def load_semi_supervised(n_labeled=100, cut_off=1000, seed=123456, conv=False, extra=False):
    """
    Load the SVHN dataset where only a fraction of data points are labeled. The amount
    of labeled data will be evenly distributed accross classes.
    :param n_labeled: Number of labeled data points.
    :param cut_off: A cut off constant so that the data set is divisable.
    :param seed: The seed for the pseudo random shuffle of data points.
    :param conv: Boolean whether the images should be vectorized or not.
    :param extra: Include the extra set or not.
    :return: Train set unlabeled and labeled, test set, validation set.
    """

    rng = np.random.RandomState(seed=seed)
    train_set, test_set, valid_set = _download(extra)

    # Create the labeled and unlabeled data evenly distributed across classes.
    x_l, y_l, x_u, y_u = create_semi_supervised(train_set, n_labeled, rng)

    train_set = (x_u, y_u)
    train_set_labeled = (x_l, y_l)
    train_x, train_t = train_set

    # shuffle data
    train_collect = np.append(train_x, train_t, axis=1)
    rng.shuffle(train_collect)
    train_set = (train_collect[:, :-10], train_collect[:, -10:])

    train_set = cut_off_dataset(train_set, cut_off, rng)

    test_set = pad_targets(test_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    if conv:
        train_set = _gen_conv(train_set)
        test_set = _gen_conv(test_set)
        if valid_set is not None:
            valid_set = _gen_conv(valid_set)

    return train_set, train_set_labeled, test_set, valid_set

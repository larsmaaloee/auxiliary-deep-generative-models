import os
import numpy as np
from utils import env_paths
from parmesan.datasets import load_mnist_realval
from data_helper import create_semi_supervised, pad_targets


def _download():
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    data = load_mnist_realval(os.path.join(env_paths.get_data_path("mnist"), "mnist.pkl.gz"))
    train_x, train_t, valid_x, valid_t, test_x, test_t = data
    return (train_x, train_t), (test_x, test_t), (valid_x, valid_t)


def load_supervised(filter_std=0.1, train_valid_combine=False):
    """
    Load the MNIST dataset.
    :param filter_std: The standard deviation threshold for keeping features.
    :param train_valid_combine: If the train set and validation set should be combined.
    :return: The train, test and validation sets.
    """
    train_set, test_set, valid_set = _download()

    if train_valid_combine:
        train_set = np.append(train_set[0], valid_set[0], axis=0), np.append(train_set[1], valid_set[1], axis=0)

    # Filter out the features with a low standard deviation.
    if filter_std > .0:
        train_x, train_t = train_set
        idx_keep = np.std(train_x, axis=0) > filter_std
        train_x = train_x[:, idx_keep]
        valid_set = (valid_set[0][:, idx_keep], valid_set[1])
        test_set = (test_set[0][:, idx_keep], test_set[1])
        train_set = (train_x, train_t)

    test_set = pad_targets(test_set)
    valid_set = pad_targets(valid_set)
    train_set = pad_targets(train_set)

    return train_set, test_set, valid_set


def load_semi_supervised(n_labeled=100, filter_std=0.1, seed=123456, train_valid_combine=False):
    """
    Load the MNIST dataset where only a fraction of data points are labeled. The amount
    of labeled data will be evenly distributed accross classes.
    :param n_labeled: Number of labeled data points.
    :param filter_std: The standard deviation threshold for keeping features.
    :param seed: The seed for the pseudo random shuffle of data points.
    :param train_valid_combine: If the train set and validation set should be combined.
    :return: Train set unlabeled and labeled, test set, validation set.
    """

    train_set, test_set, valid_set = _download()

    # Combine the train set and validation set.
    if train_valid_combine:
        train_set = np.append(train_set[0], valid_set[0], axis=0), np.append(train_set[1], valid_set[1], axis=0)

    rng = np.random.RandomState(seed=seed)

    # Create the labeled and unlabeled data evenly distributed across classes.
    x_l, y_l, x_u, y_u = create_semi_supervised(train_set, n_labeled, rng)

    # Filter out the features with a low standard deviation.
    if filter_std > .0:
        idx_keep = np.std(x_u, axis=0) > filter_std
        x_l, x_u = x_l[:, idx_keep], x_u[:, idx_keep]
        valid_set = (valid_set[0][:, idx_keep], valid_set[1])
        test_set = (test_set[0][:, idx_keep], test_set[1])

    train_set = (x_u, y_u)
    train_set_labeled = (x_l, y_l)

    # shuffle data
    train_x, train_t = train_set
    train_collect = np.append(train_x, train_t, axis=1)
    rng.shuffle(train_collect)
    train_set = (train_collect[:, :-10], train_collect[:, -10:])

    test_set = pad_targets(test_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    return train_set, train_set_labeled, test_set, valid_set

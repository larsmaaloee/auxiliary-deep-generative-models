import os
import numpy as np
from utils import env_paths
import gzip
from pylearn2.datasets import norb
from scipy.misc import imresize
from data_helper import create_semi_supervised, pad_targets, cut_off_dataset


def _download(normalize=True):
    """
    Download the NORB dataset if it is not present.
    :return: The train, test and validation set.
    """

    def load_data(data_file):
        # set temp environ data path for pylearn2.
        os.environ['PYLEARN2_DATA_PATH'] = env_paths.get_data_path("norb")

        data_dir = os.path.join(os.environ['PYLEARN2_DATA_PATH'], 'norb_small', 'original')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        dataset = os.path.join(data_dir, data_file)

        if (not os.path.isfile(dataset)):
            import urllib
            origin = (
                os.path.join('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/', data_file)
            )
            print 'Downloading data from %s' % origin

            urllib.urlretrieve(origin, dataset)
        return dataset

    def unzip(path):
        with gzip.open(path, 'rb') as infile:
            with open(path.replace('.gz', ''), 'w') as outfile:
                for line in infile:
                    outfile.write(line)

    def norm(x):
        orig_shape = (96, 96)
        new_shape = (32, 32)
        x = x.reshape((-1, 2, 96 * 96))

        def reshape_digits(x, shape):
            def rebin(_a, shape):
                img = imresize(_a, shape, interp='nearest')
                return img.reshape(-1)

            nrows = x.shape[0]
            ncols = shape[0] * shape[1]
            result = np.zeros((nrows, x.shape[1], ncols))
            for i in range(nrows):
                result[i, 0, :] = rebin(x[i, 0, :].reshape(orig_shape), shape).reshape((1, ncols))
                result[i, 1, :] = rebin(x[i, 1, :].reshape(orig_shape), shape).reshape((1, ncols))
            return result

        x = reshape_digits(x, new_shape)
        x = x.reshape((-1, 2 * np.prod(new_shape)))
        x += np.random.uniform(0, 1, size=x.shape).astype('float32')  # Add uniform noise
        x /= 256.
        x -= x.mean(axis=0)

        x = np.asarray(x, dtype='float32')
        return x

    unzip(load_data("smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz"))
    unzip(load_data("smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz"))

    train_norb = norb.SmallNORB('train')
    train_x = train_norb.X
    train_t = train_norb.y

    unzip(load_data("smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz"))
    unzip(load_data("smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz"))

    test_norb = norb.SmallNORB('test')
    test_x = test_norb.X
    test_t = test_norb.y

    if normalize:
        test_x = norm(test_x)
        train_x = norm(train_x)

    # Dummy validation set. NOTE: still in training set.
    idx = np.random.randint(0, train_x.shape[0] - 1, 5000)
    valid_x = train_x[idx, :]
    valid_t = train_t[idx]

    return (train_x, train_t), (test_x, test_t), (valid_x, valid_t)


def load_semi_supervised(n_labeled=100, cut_off=100, seed=123456, expand_channels=False, remove_channels=False):
    """
    Load the NORB dataset where only a fraction of data points are labeled. The amount
    of labeled data will be evenly distributed accross classes.
    :param n_labeled: Number of labeled data points.
    :param cut_off: A cut off constant so that the data set is divisable.
    :param seed: The seed for the pseudo random shuffle of data points.
    :param expand_channels: The pairwise images are rolled out to one dataset.
    :param remove_channels: The second image in each pair is removed.
    :return: Train set unlabeled and labeled, test set, validation set.
    """

    rng = np.random.RandomState(seed=seed)
    train_set, test_set, valid_set = _download()

    # Create the labeled and unlabeled data evenly distributed across classes.
    x_l, y_l, x_u, y_u = create_semi_supervised(train_set, n_labeled, rng)

    if expand_channels:
        x_l = x_l.reshape((-1, 2, x_l.shape[1] / 2.))
        x_l_1 = x_l[:, 0, :]
        x_l_2 = x_l[:, 1, :]
        x_l = np.append(x_l_1, x_l_2, axis=0)
        y_l = np.append(y_l, y_l, axis=0)
        x_u = x_u.reshape((-1, 2, x_u.shape[1] / 2.))
        x_u_1 = x_u[:, 0, :]
        x_u_2 = x_u[:, 1, :]
        x_u = np.append(x_u_1, x_u_2, axis=0)
        y_u = np.append(y_u, y_u, axis=0)

        test_x, test_t = test_set
        test_x = test_x.reshape((-1, 2, test_x.shape[1] / 2.))
        test_x = np.append(test_x[:, 0, :], test_x[:, 1, :], axis=0)
        test_t = np.append(test_t, test_t, axis=0)
        test_set = (test_x, test_t)

        valid_x, valid_t = valid_set
        valid_x = valid_x.reshape((-1, 2, valid_x.shape[1] / 2.))
        valid_x = np.append(valid_x[:, 0, :], valid_x[:, 1, :], axis=0)
        valid_t = np.append(valid_t, valid_t, axis=0)
        valid_set = (valid_x, valid_t)

    elif remove_channels:
        x_l = x_l.reshape((-1, 2, x_l.shape[1] / 2.))
        x_l = x_l[:, 0, :].reshape((-1, x_l.shape[-1]))
        x_u = x_u.reshape((-1, 2, x_u.shape[1] / 2.))
        x_u = x_u[:, 0, :].reshape((-1, x_u.shape[-1]))

        test_x, test_t = test_set
        test_x = test_x.reshape((-1, 2, test_x.shape[1] / 2.))
        test_x = test_x[:, 0, :].reshape((-1, test_x.shape[-1]))
        test_set = (test_x, test_t)

        valid_x, valid_t = valid_set
        valid_x = valid_x.reshape((-1, 2, valid_x.shape[1] / 2.))
        valid_x = valid_x[:, 0, :].reshape((-1, valid_x.shape[-1]))
        valid_set = (valid_x, valid_t)

    train_set = (x_u, y_u)
    train_set_labeled = (x_l, y_l)
    train_x, train_t = train_set

    # shuffle data
    train_collect = np.append(train_x, train_t, axis=1)
    rng.shuffle(train_collect)
    train_set = (train_collect[:, :-5], train_collect[:, -5:])

    train_set = cut_off_dataset(train_set, cut_off, rng)

    test_set = pad_targets(test_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    return train_set, train_set_labeled, test_set, valid_set

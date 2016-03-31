import numpy as np


def pad_targets(xy):
    """
    Pad the targets to be 1hot.
    :param xy: A tuple containing the x and y matrices.
    :return: The 1hot coded dataset.
    """
    x, y = xy
    classes = np.max(y) + 1
    tmp_data_y = np.zeros((x.shape[0], classes))
    for i, dp in zip(range(len(y)), y):
        r = np.zeros(classes)
        r[dp] = 1
        tmp_data_y[i] = r
    y = tmp_data_y
    return x, y


def cut_off_dataset(xy, cut_off, rng):
    x, y = xy
    n = x.shape[0]
    keep = n - (n % cut_off)
    idx = rng.choice(n, size=keep, replace=False)
    return x[idx, :], y[idx]


def create_semi_supervised(xy, n_labeled, rng):
    """
    Divide the dataset into labeled and unlabeled data.
    :param xy: The training set of the mnist data.
    :param n_labeled: The number of labeled data points.
    :param rng: NumPy random generator.
    :return: labeled x, labeled y, unlabeled x, unlabeled y.
    """
    x, y = xy
    n_classes = int(np.max(y) + 1)

    def _split_by_class(x, y, n_c):
        x, y = x.T, y.T
        result_x = [0] * n_c
        result_y = [0] * n_c
        for i in range(n_c):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:, idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    x, y = _split_by_class(x, y, n_classes)

    def pad_targets(y, n_c):
        new_y = np.zeros((n_c, y.shape[0]))
        for i in range(y.shape[0]):
            new_y[y[i], i] = 1
        return new_y

    for i in range(n_classes):
        y[i] = pad_targets(y[i], n_classes)

    if n_labeled % n_classes != 0:
        raise "n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)"
    n_labels_per_class = n_labeled / n_classes
    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    for i in range(n_classes):
        idx = range(x[i].shape[1])
        rng.shuffle(idx)
        x_labeled[i] = x[i][:, idx[:n_labels_per_class]]
        y_labeled[i] = y[i][:, idx[:n_labels_per_class]]
        x_unlabeled[i] = x[i]
        y_unlabeled[i] = y[i]
    return np.hstack(x_labeled).T, np.hstack(y_labeled).T, np.hstack(x_unlabeled).T, np.hstack(y_unlabeled).T

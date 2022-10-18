import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classifiers import NMC

from sklearn.metrics import pairwise_distances
import time


def load_mnist_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    y = data[:, 0]
    x = data[:, 1:] / 255
    return x, y


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape), cmap='gray')


def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_digit(x[i, :])
        if y is not None:
            plt.title('Label: ' + str(y[i]))


def split_data(x, y, tr_fraction=0.5):
    num_samples = y.size
    num_tr = int(tr_fraction * num_samples)
    num_ts = num_samples - num_tr
    tr_idx = np.zeros(shape=(num_samples,))
    tr_idx[0:num_tr] = 1

    np.random.shuffle(tr_idx)
    ytr = y[tr_idx == 1]
    xtr = x[tr_idx == 1, :]

    yts = y[tr_idx == 0]
    xts = x[tr_idx == 0, :]

    return xtr, ytr, xts, yts


# measure test error
def test_error(y_pred, yts):
    return (y_pred != yts).mean()


filename = '../data/mnist_data.csv'
x, y = load_mnist_data(filename)

clf = NMC()

print(clf.centroids)

xtr, ytr, xts, yts = split_data(x, y)
clf.fit(xtr, ytr)
print(clf.centroids)

y_pred = clf.predict(xts)
ts_error = test_error(y_pred, yts)

print(ts_error)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classifiers import NMC
from data_loaders import CDataLoaderMNIST
from data_perturb import CDataPerturbRandom


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape), cmap='gray')


def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_digit(x[i, :])
        if y is not None:
            plt.title('Label: ' + str(y[i]))


data_loader = CDataLoaderMNIST()
x, y = data_loader.load_data()

plt.figure()
plot_ten_digits(x, y)
plt.show()

data_perturb = CDataPerturbRandom(K=500)
xp = data_perturb.perturb_dataset(x)

plt.figure()
plot_ten_digits(xp, y)
plt.show()

from data_loaders import CDataLoader
import numpy as np
import pandas as pd


class CDataLoaderMNIST(CDataLoader):
    """
    Loader for the MNIST handwritten digit data
    """
    def __init__(self, filename="../data/mnist_data.csv"):
        self.filename = filename
        self._height = 28
        self._width = 28

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise ValueError("Value is not a valid string/filename!")
        self._filename = value

    def load_data(self):
        data = pd.read_csv(self.filename)
        data = np.array(data)
        y = data[:, 0]
        x = data[:, 1:] / 255
        return x, y

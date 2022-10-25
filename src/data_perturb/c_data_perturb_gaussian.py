import numpy as np
from data_perturb import CDataPerturb


class CDataPerturbGaussian(CDataPerturb):

    def __init__(self, sigma=1, min_value=0, max_value=1):
        self.sigma = sigma
        self.min_value = min_value
        self.max_value = max_value

    @property
    def sigma(self):
        return self._sigma

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @sigma.setter
    def sigma(self, val):
        self._sigma = val

    @min_value.setter
    def min_value(self, val):
        self._min_value = val

    @max_value.setter
    def max_value(self, val):
        self._max_value = val

    def data_perturbation(self, x):
        z = self.sigma * np.random.randn(x.size)  # sampling from N(0, sigma^2)
        z += x
        # x = (-1, -2, 0, 0.5, 1, 1.5) --> T T F F F F ...
        z[z < self.min_value] = self.min_value
        z[z > self.max_value] = self.max_value
        return z



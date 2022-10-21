import numpy as np
from data_perturb import CDataPerturb


class CDataPerturbRandom(CDataPerturb):

    def __init__(self, K=100, min_value=0, max_value=1):
        self.K = K  # this calls the corresponding setter
        self.min_value = min_value
        self.max_value = max_value

    @property
    def K(self):
        return self._K

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @K.setter
    def K(self, val):
        if val < 0:
            raise ValueError("Value is negative!")
        self._K = int(val)

    @min_value.setter
    def min_value(self, val):
        self._min_value = float(val)

    @max_value.setter
    def max_value(self, val):
        self._max_value = float(val)

    def data_perturbation(self, x):
        k = min(self.K, x.size)
        idx = np.array(range(x.size))
        np.random.shuffle(idx)
        idx = idx[:k]
        x[idx] = (self.max_value - self.min_value) * np.random.rand(k) + self.min_value
        return x

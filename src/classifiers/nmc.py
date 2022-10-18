import numpy as np
from sklearn.metrics import pairwise_distances


class NMC:

    def __init__(self):
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    def fit(self, xtr, ytr):
        num_classes = np.unique(ytr).size
        num_features = xtr.shape[1]
        self._centroids = np.zeros(shape=(num_classes, num_features))
        for k in range(num_classes):
            xk = xtr[ytr == k, :]
            self._centroids[k, :] = np.mean(xk, axis=0)
        return self

    def predict(self, xts):
        if self._centroids is None:  # the classifier is not trained
            raise ValueError("Train classifier first!")

        dist = pairwise_distances(xts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred

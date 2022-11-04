import unittest
import numpy as np

from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        self.x = np.zeros(shape=(10, 2))
        self.y = np.zeros(shape=(10,))
        self.clf = NMC()

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        out = self.clf.centroids
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(out.sum() == 0)

    def test_predict(self):
        self.assertRaises(ValueError, self.clf.predict, self.x)
        self.clf.fit(self.x, self.y)
        ypred = self.clf.predict(self.x)
        self.assertTrue(ypred.sum() == 0)

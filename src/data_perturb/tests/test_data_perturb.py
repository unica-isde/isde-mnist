import unittest
import numpy as np
from data_perturb import CDataPerturb


class TestDataPerturb(unittest.TestCase):

    def test_data_perturbation(self) -> None:
        self.assertRaises(TypeError, CDataPerturb)

        class Child(CDataPerturb):
            def data_perturbation(self,x):
                super().data_perturbation(x)

        self.assertRaises(NotImplementedError,
                          Child().data_perturbation, x=None)

    def test_perturb_dataset(self):
        class Child(CDataPerturb):
            def data_perturbation(self,x):
                return x

        X = np.zeros(shape=(100, 100),dtype=int)
        Z=Child().perturb_dataset(X)
        self.assertEqual(Z.shape, X.shape)
        self.assertEqual(Z.sum(),0)
        self.assertEqual(X.sum(), 0)

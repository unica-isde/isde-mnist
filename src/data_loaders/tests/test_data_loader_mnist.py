import unittest
import numpy as np

from data_loaders import CDataLoaderMNIST


class TestDataLoaderMNIST(unittest.TestCase):

    def setUp(self) -> None:
        self.data_loader = CDataLoaderMNIST()

    def test_filename(self):
        fname = 'prdkjndskjvova'
        self.data_loader.filename = fname
        self.assertEqual(self.data_loader.filename, fname)

        fname = 345  # not a string
        with self.assertRaises(ValueError):
            self.data_loader.filename = fname

        try:
            self.data_loader.filename = fname
        except ValueError:
            self.assertTrue(True)


    def test_height_width(self):
        self.assertTrue(self.data_loader.height == 28)
        self.assertTrue(self.data_loader.width == 28)

    def test_load_data(self):
        try:
            self.data_loader.filename = "../../../data/mnist_data.csv"
            x, y = self.data_loader.load_data()
        except:
            self.data_loader.filename = "data/mnist_data.csv"
            x, y = self.data_loader.load_data()
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(x, np.ndarray)
        self.assertTrue(x.size > 0)
        self.assertTrue(y.size > 0)


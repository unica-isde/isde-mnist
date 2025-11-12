import unittest
from data_loaders import CDataLoader


class TestDataLoader(unittest.TestCase):

    def test_data_loader(self) -> None:
        self.assertRaises(TypeError, CDataLoader)

        class Child(CDataLoader):
            def load_data(self):
                super().load_data()

        self.assertRaises(NotImplementedError, Child().load_data)


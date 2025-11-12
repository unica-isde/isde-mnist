import unittest
import numpy as np

import unittest
from unittest.mock import patch
from data_perturb import CDataPerturbGaussian


class TestCDataPerturbGaussian(unittest.TestCase):

    def setUp(self):
        self.default_sigma = 1.0
        self.default_min = 0.0
        self.default_max = 1.0
        self.perturb = CDataPerturbGaussian(
            sigma=self.default_sigma,
            min_value=self.default_min,
            max_value=self.default_max
        )

    def test_initialization(self):
        """Test that initialization sets parameters correctly."""
        self.assertEqual(self.perturb.sigma, self.default_sigma)
        self.assertEqual(self.perturb.min_value, self.default_min)
        self.assertEqual(self.perturb.max_value, self.default_max)

    def test_property_setters_and_getters(self):
        """Test that the getters and setters for sigma, min_value, and max_value work."""
        self.perturb.sigma = 0.5
        self.perturb.min_value = -1
        self.perturb.max_value = 2
        self.assertEqual(self.perturb.sigma, 0.5)
        self.assertEqual(self.perturb.min_value, -1)
        self.assertEqual(self.perturb.max_value, 2)

    @patch("numpy.random.randn")
    def test_data_perturbation_adds_noise(self, mock_randn):
        """Test that Gaussian noise is added correctly and values are clipped."""
        x = np.array([0.2, 0.5, 0.8])
        mock_randn.return_value = np.array([0.1, -0.5, 0.5])  # deterministic noise

        z = self.perturb.data_perturbation(x)

        # Expected: z = x + sigma * noise = [0.3, 0.0, 1.0], clipped to [0, 1]
        expected = np.array([0.3, 0.0, 1.0])
        np.testing.assert_array_almost_equal(z, expected)

    @patch("numpy.random.randn")
    def test_data_perturbation_with_different_sigma(self, mock_randn):
        """Test that changing sigma scales the noise."""
        self.perturb.sigma = 2.0
        x = np.array([0.5])
        mock_randn.return_value = np.array([0.5])  # deterministic noise

        z = self.perturb.data_perturbation(x)
        expected = np.array([0.5 + 2 * 0.5])  # = 1.5, clipped to 1.0
        np.testing.assert_array_almost_equal(z, np.array([1.0]))

    @patch("numpy.random.randn")
    def test_data_perturbation_clipping(self, mock_randn):
        """Test that output is clipped within [min_value, max_value] deterministically."""
        x = np.array([0.0, 1.0])
        self.perturb.sigma = 10  # large sigma to push values out of range

        # Mock noise values that would push one below min and one above max
        mock_randn.return_value = np.array([-1.0, 2.0])

        z = self.perturb.data_perturbation(x)

        # Expected:
        #   z = [0 + 10 * (-1), 1 + 10 * 2] = [-10, 21]
        #   After clipping -> [0, 1]
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_equal(z, expected)
        self.assertTrue(np.all(z >= self.perturb.min_value))
        self.assertTrue(np.all(z <= self.perturb.max_value))


import unittest
import numpy as np
from unittest.mock import patch

from data_perturb import CDataPerturbRandom


class TestCDataPerturbRandom(unittest.TestCase):

    def setUp(self):
        self.default_K = 3
        self.default_min = 0.0
        self.default_max = 1.0
        self.perturb = CDataPerturbRandom(
            K=self.default_K,
            min_value=self.default_min,
            max_value=self.default_max
        )

    def test_initialization(self):
        """Test that initialization sets parameters correctly."""
        self.assertEqual(self.perturb.K, self.default_K)
        self.assertEqual(self.perturb.min_value, self.default_min)
        self.assertEqual(self.perturb.max_value, self.default_max)

    def test_property_setters_and_getters(self):
        """Test that the getters and setters work as expected."""
        self.perturb.K = 10
        self.perturb.min_value = -1
        self.perturb.max_value = 2
        self.assertEqual(self.perturb.K, 10)
        self.assertEqual(self.perturb.min_value, -1.0)
        self.assertEqual(self.perturb.max_value, 2.0)

    def test_negative_K_raises_value_error(self):
        """Test that setting K < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.perturb.K = -5

    @patch("numpy.random.shuffle")
    @patch("numpy.random.rand")
    def test_data_perturbation_replaces_k_values(self, mock_rand, mock_shuffle):
        """Test that exactly K elements are perturbed with deterministic values."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.perturb.K = 2
        self.perturb.min_value = 0.0
        self.perturb.max_value = 1.0

        # Mock shuffle to produce a predictable permutation
        mock_shuffle.side_effect = lambda arr: arr.__setitem__(slice(None), np.array([4, 2, 3, 1, 0]))

        # Mock random.rand to return deterministic noise
        mock_rand.return_value = np.array([0.5, 0.8])

        z = self.perturb.data_perturbation(x.copy())

        # After shuffle, idx[:K] = [4, 2]
        # These two positions are replaced by [0.5, 0.8]
        expected = np.array([0.1, 0.2, 0.8, 0.4, 0.5])
        np.testing.assert_array_almost_equal(z, expected)

    @patch("numpy.random.shuffle")
    @patch("numpy.random.rand")
    def test_data_perturbation_respects_bounds(self, mock_rand, mock_shuffle):
        """Test that generated values are within [min_value, max_value]."""
        x = np.array([0.0, 0.0, 0.0])
        self.perturb.K = 3
        self.perturb.min_value = -2.0
        self.perturb.max_value = 2.0

        # Mock shuffle to leave order unchanged
        mock_shuffle.side_effect = lambda arr: arr
        # Mock random values between 0 and 1
        mock_rand.return_value = np.array([0.1, 0.5, 0.9])

        z = self.perturb.data_perturbation(x.copy())

        # Expected mapping: val = (max - min) * rand + min = 4*rand - 2
        expected = 4 * np.array([0.1, 0.5, 0.9]) - 2
        np.testing.assert_array_almost_equal(z, expected)

        # Check that all values are within the allowed range
        self.assertTrue(np.all(z >= self.perturb.min_value))
        self.assertTrue(np.all(z <= self.perturb.max_value))

    @patch("numpy.random.shuffle")
    @patch("numpy.random.rand")
    def test_data_perturbation_with_small_x(self, mock_rand, mock_shuffle):
        """Test that when K > len(x), all elements are replaced."""
        x = np.array([0.1, 0.2])
        self.perturb.K = 5  # larger than x.size
        mock_shuffle.side_effect = lambda arr: arr
        mock_rand.return_value = np.array([0.2, 0.7])

        z = self.perturb.data_perturbation(x.copy())
        expected = np.array([0.2, 0.7])
        np.testing.assert_array_almost_equal(z, expected)
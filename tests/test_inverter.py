"""
Unit tests for MatrixInverter class.
"""

import unittest
import numpy as np
from resource_allocation import (
    MatrixInverter,
    SingularMatrixException,
    NonSquareMatrixException
)


class TestMatrixInverter(unittest.TestCase):
    """Test cases for MatrixInverter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.inverter = MatrixInverter(epsilon=1e-12)
    
    def test_identity_matrix(self):
        """Identity matrix should invert to itself."""
        I = np.eye(5)
        I_inv = self.inverter.invert_matrix(I)
        np.testing.assert_array_almost_equal(I_inv, I, decimal=10)
    
    def test_diagonal_matrix(self):
        """Diagonal matrix inverse should be reciprocal of elements."""
        D = np.diag([2.0, 3.0, 4.0, 5.0])
        D_inv = self.inverter.invert_matrix(D)
        D_inv_expected = np.diag([0.5, 1/3, 0.25, 0.2])
        np.testing.assert_array_almost_equal(D_inv, D_inv_expected, decimal=10)
    
    def test_2x2_matrix(self):
        """Test 2x2 matrix inversion."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        A_inv = self.inverter.invert_matrix(A)
        
        # Expected inverse: [[0.6, -0.2], [-0.2, 0.4]]
        A_inv_expected = np.array([[0.6, -0.2], [-0.2, 0.4]])
        np.testing.assert_array_almost_equal(A_inv, A_inv_expected, decimal=10)
        
        # Verify A @ A_inv = I
        product = np.dot(A, A_inv)
        np.testing.assert_array_almost_equal(product, np.eye(2), decimal=10)
    
    def test_3x3_matrix(self):
        """Test 3x3 matrix inversion."""
        A = np.array([
            [2.0, 1.0, 0.5],
            [1.5, 2.5, 1.0],
            [0.5, 1.0, 2.0]
        ])
        A_inv = self.inverter.invert_matrix(A)
        
        # Verify A @ A_inv ≈ I
        product = np.dot(A, A_inv)
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)
    
    def test_singular_matrix(self):
        """Singular matrix should raise exception."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # Rank deficient
        with self.assertRaises(SingularMatrixException):
            self.inverter.invert_matrix(A)
    
    def test_near_singular_matrix(self):
        """Near-singular matrix should raise exception."""
        A = np.array([[1.0, 2.0], [2.0, 4.0 + 1e-15]])
        with self.assertRaises(SingularMatrixException):
            self.inverter.invert_matrix(A)
    
    def test_non_square_matrix(self):
        """Non-square matrix should raise exception."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(NonSquareMatrixException):
            self.inverter.invert_matrix(A)
    
    def test_orthogonal_matrix(self):
        """Orthogonal matrix inverse should be its transpose."""
        # Rotation matrix
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        R_inv = self.inverter.invert_matrix(R)
        np.testing.assert_array_almost_equal(R_inv, R.T, decimal=10)
    
    def test_random_matrices(self):
        """Test inversion on random well-conditioned matrices."""
        np.random.seed(42)
        
        for n in [5, 10, 20]:
            A = np.random.rand(n, n) + n * np.eye(n)  # Well-conditioned
            A_inv = self.inverter.invert_matrix(A)
            
            # Verify inversion
            product = np.dot(A, A_inv)
            np.testing.assert_array_almost_equal(product, np.eye(n), decimal=8)
    
    def test_verify_inverse(self):
        """Test inverse verification method."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        A_inv = self.inverter.invert_matrix(A)
        
        is_valid, error = self.inverter.verify_inverse(A, A_inv)
        
        self.assertTrue(is_valid)
        self.assertLess(error, 1e-10)
    
    def test_inversion_stats(self):
        """Test that inversion statistics are collected."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        self.inverter.invert_matrix(A)
        
        stats = self.inverter.get_inversion_stats()
        
        self.assertIn('pivots', stats)
        self.assertIn('swaps', stats)
        self.assertEqual(stats['pivots'], 2)
    
    def test_symmetric_positive_definite(self):
        """Test SPD matrix inversion."""
        # Construct SPD matrix
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_inv = self.inverter.invert_matrix(A)
        
        # Verify
        product = np.dot(A, A_inv)
        np.testing.assert_array_almost_equal(product, np.eye(2), decimal=10)
        
        # Inverse should also be SPD
        eigenvalues = np.linalg.eigvals(A_inv)
        self.assertTrue(np.all(eigenvalues > 0))


if __name__ == '__main__':
    unittest.main()

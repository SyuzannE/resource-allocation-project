"""
Unit tests for AllocationSolver class.
"""

import unittest
import numpy as np
from resource_allocation import (
    AllocationSolver,
    InvalidMatrixException,
    DimensionMismatchException
)


class TestAllocationSolver(unittest.TestCase):
    """Test cases for AllocationSolver."""
    
    def test_basic_allocation(self):
        """Test basic allocation computation."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        
        solver = AllocationSolver(A)
        x = solver.solve(b)
        
        # Verify Ax = b
        result = np.dot(A, x)
        np.testing.assert_array_almost_equal(result, b, decimal=10)
    
    def test_3x3_allocation(self):
        """Test allocation on 3x3 system."""
        A = np.array([
            [2.0, 1.0, 0.5],
            [1.5, 2.5, 1.0],
            [0.5, 1.0, 2.0]
        ])
        b = np.array([100.0, 150.0, 80.0])
        
        solver = AllocationSolver(A)
        x = solver.solve(b)
        
        # Verify solution
        result = np.dot(A, x)
        np.testing.assert_array_almost_equal(result, b, decimal=8)
    
    def test_batch_allocation(self):
        """Test batch allocation computation."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        B = np.array([
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0]
        ])
        
        solver = AllocationSolver(A)
        X = solver.solve_batch(B)
        
        # Verify each column
        for i in range(B.shape[1]):
            result = np.dot(A, X[:, i])
            np.testing.assert_array_almost_equal(result, B[:, i], decimal=10)
    
    def test_singular_matrix_rejection(self):
        """Singular matrix should be rejected during initialization."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        
        with self.assertRaises(InvalidMatrixException):
            AllocationSolver(A)
    
    def test_dimension_mismatch_single(self):
        """Wrong dimension demand vector should raise exception."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = AllocationSolver(A)
        
        wrong_b = np.array([5.0, 7.0, 9.0])  # 3 elements instead of 2
        
        with self.assertRaises(DimensionMismatchException):
            solver.solve(wrong_b)
    
    def test_dimension_mismatch_batch(self):
        """Wrong dimension batch should raise exception."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = AllocationSolver(A)
        
        wrong_B = np.array([
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0]
        ])  # 3 rows instead of 2
        
        with self.assertRaises(DimensionMismatchException):
            solver.solve_batch(wrong_B)
    
    def test_zero_demand_vector(self):
        """Zero demand vector should produce zero allocation."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([0.0, 0.0])
        
        solver = AllocationSolver(A)
        x = solver.solve(b)
        
        np.testing.assert_array_almost_equal(x, np.zeros(2), decimal=10)
    
    def test_solve_count_tracking(self):
        """Test that solve counts are tracked correctly."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = AllocationSolver(A)
        
        # Initial counts
        diag = solver.get_diagnostics()
        self.assertEqual(diag['total_solves'], 0)
        self.assertEqual(diag['batch_solves'], 0)
        
        # Single solve
        solver.solve(np.array([5.0, 7.0]))
        diag = solver.get_diagnostics()
        self.assertEqual(diag['total_solves'], 1)
        
        # Batch solve
        B = np.random.rand(2, 5)
        solver.solve_batch(B)
        diag = solver.get_diagnostics()
        self.assertEqual(diag['total_solves'], 6)  # 1 + 5
        self.assertEqual(diag['batch_solves'], 1)
    
    def test_get_matrices(self):
        """Test getting original and inverse matrices."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = AllocationSolver(A)
        
        A_retrieved = solver.get_original_matrix()
        A_inv_retrieved = solver.get_inverse_matrix()
        
        # Verify original matrix
        np.testing.assert_array_equal(A_retrieved, A)
        
        # Verify inverse
        product = np.dot(A, A_inv_retrieved)
        np.testing.assert_array_almost_equal(product, np.eye(2), decimal=10)
    
    def test_verify_solution(self):
        """Test solution verification."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        
        solver = AllocationSolver(A)
        x = solver.solve(b)
        
        is_valid, residual = solver.verify_solution(b, x)
        
        self.assertTrue(is_valid)
        self.assertLess(residual, 1e-10)
    
    def test_list_input(self):
        """Test that list inputs are accepted."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b_list = [5.0, 7.0]
        
        solver = AllocationSolver(A)
        x = solver.solve(b_list)
        
        # Should work the same as numpy array
        result = np.dot(A, x)
        np.testing.assert_array_almost_equal(result, b_list, decimal=10)
    
    def test_large_system(self):
        """Test on larger system (50x50)."""
        np.random.seed(42)
        n = 50
        
        # Create well-conditioned matrix
        A = np.random.rand(n, n) + 5 * np.eye(n)
        b = np.random.rand(n)
        
        solver = AllocationSolver(A)
        x = solver.solve(b)
        
        # Verify solution
        result = np.dot(A, x)
        np.testing.assert_array_almost_equal(result, b, decimal=6)
    
    def test_estimate_query_time(self):
        """Test query time estimation."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = AllocationSolver(A)
        
        estimate = solver.estimate_query_time(num_queries=10)
        
        self.assertIn('queries', estimate)
        self.assertIn('speedup_factor', estimate)
        self.assertIn('break_even_queries', estimate)
        self.assertEqual(estimate['queries'], 10)
        self.assertGreater(estimate['speedup_factor'], 1)


if __name__ == '__main__':
    unittest.main()

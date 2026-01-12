"""
Matrix Inversion Module

Implements Gauss-Jordan elimination with partial pivoting for
numerically stable matrix inversion.
"""

import numpy as np
import logging
from typing import Tuple, Optional

from .exceptions import (
    NonSquareMatrixException,
    SingularMatrixException,
    NumericalInstabilityException
)


class MatrixInverter:
    """
    Matrix inversion using Gauss-Jordan elimination with partial pivoting.
    
    This class provides robust matrix inversion with:
    - Partial pivoting for numerical stability
    - Comprehensive error checking
    - Verification of inversion accuracy
    - Detailed statistics tracking
    
    Example:
        >>> import numpy as np
        >>> inverter = MatrixInverter()
        >>> A = np.array([[2, 1], [1, 3]])
        >>> A_inv = inverter.invert_matrix(A)
        >>> # Verify: A @ A_inv ≈ I
    """
    
    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize the matrix inverter.
        
        Args:
            epsilon: Numerical tolerance for zero detection
        """
        self.epsilon = epsilon
        self.stats = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def invert_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of matrix A using Gauss-Jordan elimination.
        
        Args:
            A: Square matrix to invert (n x n)
            
        Returns:
            A_inv: Inverse matrix (n x n)
            
        Raises:
            NonSquareMatrixException: If matrix is not square
            SingularMatrixException: If matrix is singular or near-singular
            
        Time Complexity: O(n³)
        """
        n = A.shape[0]
        
        # Validate square matrix
        if A.shape[1] != n:
            raise NonSquareMatrixException(
                f"Matrix must be square, got shape {A.shape}"
            )
        
        # Create augmented matrix [A | I]
        aug = np.hstack([A.astype(float), np.eye(n)])
        
        # Reset statistics
        self.stats = {'pivots': 0, 'swaps': 0, 'max_pivot': 0, 'min_pivot': float('inf')}
        
        # Gauss-Jordan elimination with partial pivoting
        for i in range(n):
            # Find pivot (row with largest absolute value in column i)
            pivot_row = self._find_pivot(aug, i, n)
            pivot_value = abs(aug[pivot_row, i])
            
            # Track pivot statistics
            self.stats['max_pivot'] = max(self.stats['max_pivot'], pivot_value)
            self.stats['min_pivot'] = min(self.stats['min_pivot'], pivot_value)
            
            # Check for singularity
            if pivot_value < self.epsilon:
                raise SingularMatrixException(
                    f"Matrix is singular or near-singular: "
                    f"pivot at column {i} has magnitude {pivot_value:.2e}"
                )
            
            # Swap rows if needed (partial pivoting)
            if pivot_row != i:
                aug[[i, pivot_row]] = aug[[pivot_row, i]]
                self.stats['swaps'] += 1
            
            self.stats['pivots'] += 1
            
            # Normalize pivot row (make pivot element = 1)
            pivot = aug[i, i]
            aug[i] = aug[i] / pivot
            
            # Eliminate all other elements in column i
            for j in range(n):
                if i != j:
                    factor = aug[j, i]
                    aug[j] = aug[j] - factor * aug[i]
        
        # Extract inverse from right half of augmented matrix
        A_inv = aug[:, n:]
        
        self.logger.info(
            f"Matrix inversion complete: "
            f"{self.stats['pivots']} pivots, {self.stats['swaps']} swaps"
        )
        
        return A_inv
    
    def _find_pivot(self, aug: np.ndarray, col: int, n: int) -> int:
        """
        Find the row with maximum absolute value in the specified column.
        
        This implements partial pivoting to improve numerical stability.
        
        Args:
            aug: Augmented matrix
            col: Column to search
            n: Matrix dimension
            
        Returns:
            Row index with maximum absolute value
        """
        return max(range(col, n), key=lambda r: abs(aug[r, col]))
    
    def verify_inverse(
        self,
        A: np.ndarray,
        A_inv: np.ndarray,
        tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Verify that A @ A_inv ≈ I.
        
        Args:
            A: Original matrix
            A_inv: Computed inverse
            tolerance: Maximum acceptable error
            
        Returns:
            Tuple of (is_valid, max_error)
        """
        # Compute A @ A_inv
        product = np.dot(A, A_inv)
        
        # Compare with identity matrix
        identity = np.eye(A.shape[0])
        error_matrix = product - identity
        
        # Compute maximum absolute error
        max_error = np.max(np.abs(error_matrix))
        
        is_valid = max_error < tolerance
        
        if not is_valid:
            self.logger.warning(
                f"Inverse verification failed: max error = {max_error:.2e} "
                f"(tolerance = {tolerance:.2e})"
            )
        
        return is_valid, max_error
    
    def get_inversion_stats(self) -> dict:
        """
        Get statistics from the last inversion operation.
        
        Returns:
            Dictionary with inversion statistics
        """
        return self.stats.copy()
    
    def estimate_inversion_error(
        self,
        A: np.ndarray,
        A_inv: np.ndarray
    ) -> dict:
        """
        Estimate various error metrics for the computed inverse.
        
        Args:
            A: Original matrix
            A_inv: Computed inverse
            
        Returns:
            Dictionary with error metrics
        """
        n = A.shape[0]
        I = np.eye(n)
        
        # Forward error: ||A @ A_inv - I||
        forward_product = np.dot(A, A_inv)
        forward_error = np.linalg.norm(forward_product - I, ord='fro')
        
        # Backward error: ||A_inv @ A - I||
        backward_product = np.dot(A_inv, A)
        backward_error = np.linalg.norm(backward_product - I, ord='fro')
        
        # Relative errors
        rel_forward = forward_error / np.linalg.norm(I, ord='fro')
        rel_backward = backward_error / np.linalg.norm(I, ord='fro')
        
        return {
            'forward_error': forward_error,
            'backward_error': backward_error,
            'relative_forward_error': rel_forward,
            'relative_backward_error': rel_backward,
            'max_element_error': max(
                np.max(np.abs(forward_product - I)),
                np.max(np.abs(backward_product - I))
            )
        }

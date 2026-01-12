"""
Allocation Solver Module

Main interface for resource allocation computations using precomputed
inverse matrices.
"""

import numpy as np
import logging
from typing import Optional, Union

from .matrix_inverter import MatrixInverter
from .invertibility_checker import InvertibilityChecker
from .exceptions import (
    InvalidMatrixException,
    DimensionMismatchException
)


class AllocationSolver:
    """
    Main allocation solver using precomputed inverse matrix.
    
    This class manages the complete allocation workflow:
    - Matrix validation and inversion
    - Single and batch allocation computation
    - Inverse matrix caching
    - Result validation
    
    Example:
        >>> import numpy as np
        >>> A = np.array([[2, 1], [1, 3]])
        >>> solver = AllocationSolver(A)
        >>> 
        >>> # Single allocation
        >>> b = np.array([5, 7])
        >>> x = solver.solve(b)
        >>> 
        >>> # Batch allocation
        >>> B = np.random.rand(2, 100)
        >>> X = solver.solve_batch(B)
    """
    
    def __init__(
        self,
        A: np.ndarray,
        verify: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the allocation solver.
        
        Args:
            A: Dependency matrix (n x n)
            verify: Whether to verify inverse correctness
            logger: Optional logger instance
            
        Raises:
            InvalidMatrixException: If matrix is not invertible
            
        Time Complexity: O(n³) for initial inversion
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Check invertibility
        checker = InvertibilityChecker()
        result = checker.check(A)
        
        if not result.is_invertible:
            raise InvalidMatrixException(
                f'Cannot invert matrix: {result.message}'
            )
        
        self.logger.info(f'Matrix is invertible: {result.message}')
        
        # Store original matrix
        self.A = A.copy()
        self.n = A.shape[0]
        self.condition_number = result.condition_number
        
        # Compute inverse
        engine = MatrixInverter()
        self.A_inv = engine.invert_matrix(A)
        
        # Verify if requested
        if verify:
            is_valid, error = engine.verify_inverse(A, self.A_inv)
            if not is_valid:
                self.logger.warning(
                    f'Inverse verification failed: max error = {error:.2e}'
                )
            else:
                self.logger.info(f'Inverse verified: max error = {error:.2e}')
        
        # Statistics
        self.solve_count = 0
        self.batch_count = 0
    
    def solve(self, b: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute allocation x = A⁻¹b for given demand vector.
        
        Args:
            b: Demand vector (n,) or list of length n
            
        Returns:
            x: Allocation vector (n,)
            
        Raises:
            DimensionMismatchException: If b has wrong dimension
            
        Time Complexity: O(n²)
        """
        # Convert to numpy array if needed
        b = np.asarray(b, dtype=float)
        
        # Validate dimensions
        if len(b) != self.n:
            raise DimensionMismatchException(
                f'Dimension mismatch: expected vector of length {self.n}, '
                f'got {len(b)}'
            )
        
        # Compute allocation: x = A_inv @ b
        x = np.dot(self.A_inv, b)
        
        self.solve_count += 1
        
        return x
    
    def solve_batch(self, B: np.ndarray) -> np.ndarray:
        """
        Compute allocations for multiple demand vectors efficiently.
        
        Args:
            B: Demand matrix (n x m) where each column is a demand vector
            
        Returns:
            X: Allocation matrix (n x m)
            
        Raises:
            DimensionMismatchException: If B has wrong dimensions
            
        Time Complexity: O(mn²) for m demand vectors
        """
        # Validate dimensions
        if B.shape[0] != self.n:
            raise DimensionMismatchException(
                f'Dimension mismatch: expected {self.n} rows, got {B.shape[0]}'
            )
        
        # Compute allocations: X = A_inv @ B (efficient matrix multiplication)
        X = np.dot(self.A_inv, B)
        
        self.batch_count += 1
        self.solve_count += B.shape[1]
        
        return X
    
    def get_diagnostics(self) -> dict:
        """
        Get solver diagnostics and statistics.
        
        Returns:
            Dictionary with diagnostic information
        """
        return {
            'matrix_size': self.n,
            'condition_number': self.condition_number,
            'total_solves': self.solve_count,
            'batch_solves': self.batch_count,
            'is_well_conditioned': self.condition_number < 1e10,
            'is_moderately_conditioned': 1e10 <= self.condition_number < 1e13,
            'is_ill_conditioned': self.condition_number >= 1e13
        }
    
    def verify_solution(
        self,
        b: np.ndarray,
        x: np.ndarray,
        tolerance: float = 1e-10
    ) -> tuple:
        """
        Verify that computed allocation satisfies Ax = b.
        
        Args:
            b: Original demand vector
            x: Computed allocation vector
            tolerance: Acceptable error tolerance
            
        Returns:
            Tuple of (is_valid, residual_norm)
        """
        # Compute residual: Ax - b
        residual = np.dot(self.A, x) - b
        residual_norm = np.linalg.norm(residual)
        
        # Compute relative residual
        b_norm = np.linalg.norm(b)
        if b_norm > 0:
            relative_residual = residual_norm / b_norm
        else:
            relative_residual = residual_norm
        
        is_valid = relative_residual < tolerance
        
        return is_valid, relative_residual
    
    def get_inverse_matrix(self) -> np.ndarray:
        """
        Get the precomputed inverse matrix.
        
        Returns:
            A_inv: Inverse matrix (n x n)
        """
        return self.A_inv.copy()
    
    def get_original_matrix(self) -> np.ndarray:
        """
        Get the original dependency matrix.
        
        Returns:
            A: Original matrix (n x n)
        """
        return self.A.copy()
    
    def estimate_query_time(self, num_queries: int = 1) -> dict:
        """
        Estimate time for given number of queries.
        
        Args:
            num_queries: Number of allocation queries
            
        Returns:
            Dictionary with time estimates (in arbitrary units)
        """
        n = self.n
        
        # Rough complexity estimates
        inversion_cost = n ** 3
        query_cost = n ** 2
        
        total_with_inversion = inversion_cost + num_queries * query_cost
        total_without_inversion = num_queries * inversion_cost
        
        return {
            'queries': num_queries,
            'cost_with_inversion': total_with_inversion,
            'cost_without_inversion': total_without_inversion,
            'speedup_factor': total_without_inversion / total_with_inversion,
            'break_even_queries': max(1, int(inversion_cost / (inversion_cost - query_cost)))
        }

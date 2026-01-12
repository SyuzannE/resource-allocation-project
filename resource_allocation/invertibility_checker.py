"""
Invertibility Checker Module

Verifies matrix invertibility and analyzes numerical conditioning.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .exceptions import NonSquareMatrixException, SingularMatrixException


class InvertibilityStatus(Enum):
    """Status enumeration for invertibility check results."""
    INVERTIBLE = "invertible"
    NON_SQUARE = "non_square"
    SINGULAR = "singular"
    ILL_CONDITIONED = "ill_conditioned"


@dataclass
class InvertibilityResult:
    """
    Result of invertibility check.
    
    Attributes:
        status: InvertibilityStatus indicating the result
        is_invertible: Boolean indicating if matrix is invertible
        condition_number: Condition number of the matrix
        determinant: Determinant value (None if not square)
        message: Descriptive message about the result
    """
    status: InvertibilityStatus
    is_invertible: bool
    condition_number: float
    determinant: Optional[float]
    message: str


class InvertibilityChecker:
    """
    Checks matrix invertibility and analyzes numerical conditioning.
    
    This class provides comprehensive checks including:
    - Square matrix verification
    - Determinant computation
    - Condition number analysis
    - Singularity detection
    
    Example:
        >>> import numpy as np
        >>> checker = InvertibilityChecker()
        >>> A = np.array([[2, 1], [1, 3]])
        >>> result = checker.check(A)
        >>> print(result.is_invertible)
        True
    """
    
    def __init__(
        self,
        det_threshold: float = 1e-10,
        cond_threshold: float = 1e13
    ):
        """
        Initialize the invertibility checker.
        
        Args:
            det_threshold: Minimum absolute determinant value for non-singularity
            cond_threshold: Maximum condition number for well-conditioned matrices
        """
        self.det_threshold = det_threshold
        self.cond_threshold = cond_threshold
    
    def check(self, A: np.ndarray) -> InvertibilityResult:
        """
        Perform comprehensive invertibility check.
        
        Args:
            A: Matrix to check (n x n)
            
        Returns:
            InvertibilityResult with detailed status
        """
        # Check if square
        if A.shape[0] != A.shape[1]:
            return InvertibilityResult(
                status=InvertibilityStatus.NON_SQUARE,
                is_invertible=False,
                condition_number=float("inf"),
                determinant=None,
                message=f"Matrix is not square: {A.shape}"
            )
        
        # Compute determinant
        det = np.linalg.det(A)
        
        # Check for singularity
        if abs(det) < self.det_threshold:
            return InvertibilityResult(
                status=InvertibilityStatus.SINGULAR,
                is_invertible=False,
                condition_number=float("inf"),
                determinant=det,
                message=f"Matrix is singular: |det| = {abs(det):.2e}"
            )
        
        # Compute condition number
        cond = np.linalg.cond(A)
        
        # Check conditioning
        if cond > self.cond_threshold:
            return InvertibilityResult(
                status=InvertibilityStatus.ILL_CONDITIONED,
                is_invertible=False,
                condition_number=cond,
                determinant=det,
                message=f"Matrix is ill-conditioned: κ = {cond:.2e}"
            )
        
        # Matrix is invertible
        return InvertibilityResult(
            status=InvertibilityStatus.INVERTIBLE,
            is_invertible=True,
            condition_number=cond,
            determinant=det,
            message=f"Matrix is well-conditioned: κ = {cond:.2e}"
        )
    
    def compute_condition_number(
        self,
        A: np.ndarray,
        norm: str = "2"
    ) -> float:
        """
        Compute condition number using specified norm.
        
        Args:
            A: Input matrix
            norm: Norm type ('1', '2', 'fro', 'inf')
            
        Returns:
            Condition number κ(A)
        """
        return np.linalg.cond(A, p=norm)
    
    def is_well_conditioned(
        self,
        A: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if matrix is well-conditioned.
        
        Args:
            A: Input matrix
            threshold: Optional custom threshold (uses default if None)
            
        Returns:
            True if well-conditioned, False otherwise
        """
        threshold = threshold or self.cond_threshold
        return self.compute_condition_number(A) < threshold
    
    def get_singularity_indicators(self, A: np.ndarray) -> dict:
        """
        Get multiple indicators of singularity.
        
        Args:
            A: Input matrix
            
        Returns:
            Dictionary with various singularity indicators
        """
        det = np.linalg.det(A)
        cond = np.linalg.cond(A)
        rank = np.linalg.matrix_rank(A)
        
        return {
            'determinant': det,
            'abs_determinant': abs(det),
            'condition_number': cond,
            'rank': rank,
            'expected_rank': A.shape[0],
            'is_full_rank': rank == A.shape[0],
            'is_near_singular': abs(det) < self.det_threshold,
            'is_ill_conditioned': cond > self.cond_threshold
        }

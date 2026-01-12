"""
Stability Analyzer Module

Provides comprehensive numerical stability analysis and error estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class StabilityReport:
    """
    Comprehensive stability analysis report.
    
    Attributes:
        condition_number: Condition number κ(A)
        estimated_error_bound: Theoretical error bound
        relative_residual: Actual relative residual
        is_stable: Whether solution is numerically stable
        warnings: List of warning messages
        recommendations: List of recommendations for improvement
    """
    condition_number: float
    estimated_error_bound: float
    relative_residual: float
    is_stable: bool
    warnings: List[str]
    recommendations: List[str]


class StabilityAnalyzer:
    """
    Analyzes numerical stability and error propagation.
    
    This class provides comprehensive stability diagnostics including:
    - Condition number analysis
    - Error bound estimation
    - Residual computation
    - Method comparison
    - Recommendations for improvement
    
    Example:
        >>> analyzer = StabilityAnalyzer()
        >>> report = analyzer.analyze(A, A_inv, b)
        >>> if not report.is_stable:
        ...     print("Warnings:", report.warnings)
        ...     print("Recommendations:", report.recommendations)
    """
    
    def __init__(self):
        """Initialize the stability analyzer."""
        self.machine_epsilon = np.finfo(float).eps
    
    def analyze(
        self,
        A: np.ndarray,
        A_inv: np.ndarray,
        b: Optional[np.ndarray] = None
    ) -> StabilityReport:
        """
        Perform comprehensive stability analysis.
        
        Args:
            A: Original matrix
            A_inv: Computed inverse
            b: Optional demand vector for residual analysis
            
        Returns:
            StabilityReport with complete diagnostics
        """
        warnings = []
        recommendations = []
        
        # Compute condition number
        cond = np.linalg.cond(A)
        
        # Estimate error bound based on condition number and machine epsilon
        error_bound = cond * self.machine_epsilon
        
        # Compute relative residual if b is provided
        rel_residual = 0.0
        if b is not None:
            x = np.dot(A_inv, b)
            residual = np.dot(A, x) - b
            rel_residual = np.linalg.norm(residual) / np.linalg.norm(b)
        
        # Determine stability
        is_stable = cond < 1e10 and error_bound < 1e-6
        
        # Generate warnings based on analysis
        if cond > 1e12:
            warnings.append(
                f'Very high condition number (κ={cond:.2e}): '
                'results may be unreliable'
            )
            recommendations.append('Consider matrix regularization')
            recommendations.append('Use iterative refinement for better accuracy')
        elif cond > 1e10:
            warnings.append(
                f'High condition number (κ={cond:.2e}): '
                'some accuracy loss expected'
            )
            recommendations.append('Monitor solution quality closely')
        
        if rel_residual > 1e-6:
            warnings.append(
                f'Large relative residual ({rel_residual:.2e}): '
                'solution may be inaccurate'
            )
            recommendations.append('Verify input data quality')
        
        if error_bound > 1e-8:
            warnings.append(
                f'Error bound ({error_bound:.2e}) exceeds typical tolerance'
            )
            recommendations.append('Use iterative refinement')
        
        # Check inverse accuracy
        product = np.dot(A, A_inv)
        identity = np.eye(A.shape[0])
        inverse_error = np.max(np.abs(product - identity))
        
        if inverse_error > 1e-8:
            warnings.append(
                f'Inverse verification error ({inverse_error:.2e}) is high'
            )
            recommendations.append('Recompute inverse with higher precision')
        
        return StabilityReport(
            condition_number=cond,
            estimated_error_bound=error_bound,
            relative_residual=rel_residual,
            is_stable=is_stable,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def compare_methods(
        self,
        A: np.ndarray,
        b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare inversion-based solution with direct solving.
        
        Args:
            A: Dependency matrix
            b: Demand vector
            
        Returns:
            Dictionary comparing different solution methods
        """
        # Method 1: Matrix inversion
        A_inv = np.linalg.inv(A)
        x_inv = np.dot(A_inv, b)
        
        # Method 2: Direct solve (LU decomposition)
        x_solve = np.linalg.solve(A, b)
        
        # Method 3: Least squares (for comparison)
        x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Compare results
        diff_inv_solve = np.linalg.norm(x_inv - x_solve)
        diff_inv_lstsq = np.linalg.norm(x_inv - x_lstsq)
        
        rel_diff_inv_solve = diff_inv_solve / np.linalg.norm(x_solve)
        rel_diff_inv_lstsq = diff_inv_lstsq / np.linalg.norm(x_lstsq)
        
        # Compute residuals
        residual_inv = np.linalg.norm(np.dot(A, x_inv) - b)
        residual_solve = np.linalg.norm(np.dot(A, x_solve) - b)
        residual_lstsq = np.linalg.norm(np.dot(A, x_lstsq) - b)
        
        return {
            'inversion_solution': x_inv,
            'direct_solution': x_solve,
            'lstsq_solution': x_lstsq,
            'difference_inv_vs_solve': diff_inv_solve,
            'difference_inv_vs_lstsq': diff_inv_lstsq,
            'relative_difference_inv_vs_solve': rel_diff_inv_solve,
            'relative_difference_inv_vs_lstsq': rel_diff_inv_lstsq,
            'residual_inversion': residual_inv,
            'residual_direct': residual_solve,
            'residual_lstsq': residual_lstsq,
            'methods_agree': rel_diff_inv_solve < 1e-8,
            'recommended_method': self._recommend_method(
                residual_inv, residual_solve, residual_lstsq
            )
        }
    
    def _recommend_method(
        self,
        res_inv: float,
        res_solve: float,
        res_lstsq: float
    ) -> str:
        """Recommend best method based on residuals."""
        methods = {
            'inversion': res_inv,
            'direct_solve': res_solve,
            'least_squares': res_lstsq
        }
        return min(methods, key=methods.get)
    
    def estimate_error_bound(
        self,
        A: np.ndarray,
        b: np.ndarray,
        perturbation_level: float = 1e-10
    ) -> float:
        """
        Estimate error bound through perturbation analysis.
        
        Args:
            A: Dependency matrix
            b: Demand vector
            perturbation_level: Magnitude of perturbation
            
        Returns:
            Estimated relative error bound
        """
        # Original solution
        x = np.linalg.solve(A, b)
        
        # Perturb b slightly
        b_pert = b + perturbation_level * np.random.randn(len(b))
        x_pert = np.linalg.solve(A, b_pert)
        
        # Compute relative error
        delta_x = np.linalg.norm(x_pert - x)
        delta_b = np.linalg.norm(b_pert - b)
        
        # Error amplification factor
        error_amplification = (delta_x / np.linalg.norm(x)) / (delta_b / np.linalg.norm(b))
        
        return error_amplification
    
    def analyze_pivot_behavior(self, A: np.ndarray) -> Dict[str, Any]:
        """
        Analyze pivot behavior during Gaussian elimination.
        
        Args:
            A: Matrix to analyze
            
        Returns:
            Dictionary with pivot analysis
        """
        n = A.shape[0]
        A_copy = A.copy().astype(float)
        pivots = []
        
        # Simulate Gaussian elimination
        for i in range(n):
            # Find maximum pivot in column
            max_idx = np.argmax(np.abs(A_copy[i:, i])) + i
            pivot = A_copy[max_idx, i]
            pivots.append(abs(pivot))
            
            if abs(pivot) > 1e-15:
                # Swap rows
                if max_idx != i:
                    A_copy[[i, max_idx]] = A_copy[[max_idx, i]]
                
                # Eliminate
                for j in range(i + 1, n):
                    factor = A_copy[j, i] / A_copy[i, i]
                    A_copy[j, i:] -= factor * A_copy[i, i:]
        
        pivots = np.array(pivots)
        
        return {
            'pivots': pivots,
            'min_pivot': np.min(pivots),
            'max_pivot': np.max(pivots),
            'pivot_ratio': np.max(pivots) / np.min(pivots) if np.min(pivots) > 0 else float('inf'),
            'small_pivots': np.sum(pivots < 1e-10),
            'pivot_growth': np.max(pivots) / pivots[0] if pivots[0] > 0 else float('inf')
        }
    
    def suggest_improvements(self, A: np.ndarray) -> List[str]:
        """
        Suggest improvements for better numerical stability.
        
        Args:
            A: Matrix to analyze
            
        Returns:
            List of suggestions
        """
        suggestions = []
        cond = np.linalg.cond(A)
        
        if cond > 1e10:
            suggestions.append(
                "Consider regularization: add small diagonal term (A + λI)"
            )
            suggestions.append(
                "Try row/column scaling to improve conditioning"
            )
            suggestions.append(
                "Use SVD-based pseudoinverse for more stability"
            )
        
        # Check for small diagonal elements
        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-10):
            suggestions.append(
                "Matrix has small diagonal elements; consider pivoting strategy"
            )
        
        # Check for symmetry
        if not np.allclose(A, A.T):
            suggestions.append(
                "Matrix is not symmetric; symmetrizing may improve stability"
            )
        
        return suggestions

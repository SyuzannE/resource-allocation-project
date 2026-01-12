"""
Real-Time Resource Allocation Using Inverse Matrices

A production-ready software solution for real-time resource allocation
in cloud-based distributed systems using inverse matrix computations.

Author: Syuzanna Ghazaryan
Institution: French University in Armenia
Year: 2026
"""

__version__ = "1.0.0"
__author__ = "Syuzanna Ghazaryan"
__email__ = "syuzanna.ghazaryan@example.com"

from .matrix_inverter import MatrixInverter
from .invertibility_checker import (
    InvertibilityChecker,
    InvertibilityStatus,
    InvertibilityResult
)
from .allocation_solver import AllocationSolver
from .stability_analyzer import StabilityAnalyzer, StabilityReport
from .exceptions import (
    AllocationException,
    InvalidMatrixException,
    NonSquareMatrixException,
    SingularMatrixException,
    IllConditionedMatrixException,
    DimensionMismatchException,
    NumericalInstabilityException,
    ComputationFailedException
)

__all__ = [
    # Core classes
    'MatrixInverter',
    'InvertibilityChecker',
    'AllocationSolver',
    'StabilityAnalyzer',
    
    # Data classes
    'InvertibilityStatus',
    'InvertibilityResult',
    'StabilityReport',
    
    # Exceptions
    'AllocationException',
    'InvalidMatrixException',
    'NonSquareMatrixException',
    'SingularMatrixException',
    'IllConditionedMatrixException',
    'DimensionMismatchException',
    'NumericalInstabilityException',
    'ComputationFailedException',
]

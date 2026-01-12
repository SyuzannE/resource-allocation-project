"""
Custom exceptions for the resource allocation system.
"""


class AllocationException(Exception):
    """Base exception for all allocation-related errors."""
    pass


class InvalidMatrixException(AllocationException):
    """Raised when matrix is invalid for inversion."""
    pass


class NonSquareMatrixException(InvalidMatrixException):
    """Raised when matrix is not square."""
    pass


class SingularMatrixException(InvalidMatrixException):
    """Raised when matrix is singular (non-invertible)."""
    pass


class IllConditionedMatrixException(InvalidMatrixException):
    """Raised when matrix is ill-conditioned."""
    pass


class DimensionMismatchException(AllocationException):
    """Raised when dimensions don't match for operations."""
    pass


class NumericalInstabilityException(AllocationException):
    """Raised when numerical instability is detected."""
    pass


class ComputationFailedException(AllocationException):
    """Raised when computation fails unexpectedly."""
    pass

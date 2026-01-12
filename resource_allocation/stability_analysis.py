"""
Advanced Usage Example

Demonstrates advanced features including stability analysis,
error handling, and performance optimization.
"""

import numpy as np
from resource_allocation import (
    AllocationSolver,
    StabilityAnalyzer,
    InvalidMatrixException
)


def example_stability_analysis():
    """Demonstrate stability analysis features."""
    print("=" * 60)
    print("Example: Stability Analysis")
    print("=" * 60)
    print()
    
    # Create a moderately conditioned matrix
    A = np.array([
        [10.0, 1.0, 0.1],
        [1.0, 10.0, 1.0],
        [0.1, 1.0, 10.0]
    ])
    
    solver = AllocationSolver(A)
    
    # Get inverse
    A_inv = solver.get_inverse_matrix()
    
    # Analyze stability
    from resource_allocation import StabilityAnalyzer
    analyzer = StabilityAnalyzer()
    
    b = np.array([100, 150, 80])
    report = analyzer.analyze(A, solver.get_inverse_matrix(), b)
    
    print("Stability Analysis Report:")
    print(f"  Condition number: {report.condition_number:.2e}")
    print(f"  Error bound: {report.estimated_error_bound:.2e}")
    print(f"  Relative residual: {report.relative_residual:.2e}")
    print(f"  Is stable: {report.is_stable}")
    
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")


if __name__ == '__main__':
    main()

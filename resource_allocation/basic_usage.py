"""
Basic Usage Example

This example demonstrates the basic usage of the resource allocation system.
"""

import numpy as np
from resource_allocation import AllocationSolver


def main():
    print("=" * 60)
    print("Resource Allocation System - Basic Example")
    print("=" * 60)
    print()
    
    # Define dependency matrix
    # Rows: Resources (CPU, Memory, Bandwidth)
    # Columns: Services (Service A, B, C)
    A = np.array([
        [2.0, 1.0, 0.5],  # CPU: cores per service instance
        [1.5, 2.5, 1.0],  # Memory: GB per service instance
        [0.5, 1.0, 2.0]   # Bandwidth: Gbps per service instance
    ])
    
    print("Dependency Matrix A:")
    print(A)
    print()
    
    # Initialize solver
    print("Initializing solver...")
    solver = AllocationSolver(A, verify=True)
    print("✓ Solver initialized successfully")
    print()
    
    # Get diagnostics
    diag = solver.get_diagnostics()
    print(f"Matrix size: {diag['matrix_size']}x{diag['matrix_size']}")
    print(f"Condition number: {diag['condition_number']:.2f}")
    print(f"Well-conditioned: {diag['is_well_conditioned']}")
    print()
    
    # Example 1: Single allocation
    print("-" * 60)
    print("Example 1: Single Allocation")
    print("-" * 60)
    
    demand = np.array([100, 150, 80])  # Total CPU, Memory, Bandwidth available
    print(f"Total available resources: {demand}")
    
    allocation = solver.solve(demand)
    print(f"Service allocations: {allocation}")
    print(f"  Service A: {allocation[0]:.2f} instances")
    print(f"  Service B: {allocation[1]:.2f} instances")
    print(f"  Service C: {allocation[2]:.2f} instances")
    print()
    
    # Verify solution
    is_valid, residual = solver.verify_solution(demand, allocation)
    print(f"Solution valid: {is_valid}")
    print(f"Relative residual: {residual:.2e}")
    print()
    
    # Example 2: Multiple scenarios (batch processing)
    print("-" * 60)
    print("Example 2: Batch Allocation (5 scenarios)")
    print("-" * 60)
    
    # Generate 5 different demand scenarios
    scenarios = np.array([
        [100, 150, 80],   # Peak hours
        [80, 120, 60],    # Normal load
        [120, 180, 100],  # High load
        [60, 90, 40],     # Low load
        [150, 200, 120]   # Maximum capacity
    ]).T  # Transpose to get (3 x 5) matrix
    
    print("Demand scenarios:")
    for i in range(scenarios.shape[1]):
        print(f"  Scenario {i+1}: {scenarios[:, i]}")
    print()
    
    # Compute all allocations at once
    allocations = solver.solve_batch(scenarios)
    
    print("Computed allocations:")
    for i in range(allocations.shape[1]):
        print(f"  Scenario {i+1}: [{allocations[0, i]:.2f}, "
              f"{allocations[1, i]:.2f}, {allocations[2, i]:.2f}]")
    print()
    
    # Example 3: Performance estimation
    print("-" * 60)
    print("Example 3: Performance Analysis")
    print("-" * 60)
    
    for num_queries in [1, 10, 100, 1000]:
        estimate = solver.estimate_query_time(num_queries)
        print(f"For {num_queries} queries:")
        print(f"  Speedup factor: {estimate['speedup_factor']:.2f}x")
        print(f"  Break-even point: {estimate['break_even_queries']} queries")
        print()
    
    # Final statistics
    final_diag = solver.get_diagnostics()
    print("-" * 60)
    print("Final Statistics")
    print("-" * 60)
    print(f"Total allocations computed: {final_diag['total_solves']}")
    print(f"Batch operations: {final_diag['batch_solves']}")
    print()
    print("✓ Example completed successfully!")


if __name__ == '__main__':
    main()

# API Reference

## Core Classes

### AllocationSolver

Main interface for resource allocation computations.

#### Constructor

```python
AllocationSolver(A, verify=True, logger=None)
```

**Parameters:**
- `A` (ndarray): Dependency matrix (n × n)
- `verify` (bool): Whether to verify inverse correctness (default: True)
- `logger` (Logger): Optional logging instance

**Raises:**
- `InvalidMatrixException`: If matrix is not invertible

**Time Complexity:** O(n³) for initial inversion

#### Methods

##### solve(b)

Compute allocation x = A⁻¹b for given demand vector.

```python
x = solver.solve(b)
```

**Parameters:**
- `b` (ndarray or list): Demand vector of length n

**Returns:**
- `x` (ndarray): Allocation vector of length n

**Raises:**
- `DimensionMismatchException`: If b has wrong dimension

**Time Complexity:** O(n²)

##### solve_batch(B)

Compute allocations for multiple demand vectors efficiently.

```python
X = solver.solve_batch(B)
```

**Parameters:**
- `B` (ndarray): Demand matrix (n × m) where each column is a demand vector

**Returns:**
- `X` (ndarray): Allocation matrix (n × m)

**Raises:**
- `DimensionMismatchException`: If B has wrong dimensions

**Time Complexity:** O(mn²) for m demand vectors

##### get_diagnostics()

Get solver diagnostics and statistics.

```python
diag = solver.get_diagnostics()
```

**Returns:**
- Dictionary with diagnostic information including:
  - `matrix_size`: Dimension of the matrix
  - `condition_number`: κ(A)
  - `total_solves`: Number of allocation computations
  - `batch_solves`: Number of batch operations
  - `is_well_conditioned`: Boolean flag

##### verify_solution(b, x, tolerance=1e-10)

Verify that computed allocation satisfies Ax = b.

```python
is_valid, residual = solver.verify_solution(b, x)
```

**Parameters:**
- `b` (ndarray): Original demand vector
- `x` (ndarray): Computed allocation vector
- `tolerance` (float): Acceptable error tolerance

**Returns:**
- Tuple of (is_valid, relative_residual)

### MatrixInverter

Matrix inversion using Gauss-Jordan elimination with partial pivoting.

#### Constructor

```python
MatrixInverter(epsilon=1e-12)
```

**Parameters:**
- `epsilon` (float): Numerical tolerance for zero detection

#### Methods

##### invert_matrix(A)

Compute the inverse of matrix A.

```python
A_inv = inverter.invert_matrix(A)
```

**Parameters:**
- `A` (ndarray): Square matrix to invert (n × n)

**Returns:**
- `A_inv` (ndarray): Inverse matrix (n × n)

**Raises:**
- `NonSquareMatrixException`: If matrix is not square
- `SingularMatrixException`: If matrix is singular

**Time Complexity:** O(n³)

##### verify_inverse(A, A_inv, tolerance=1e-10)

Verify that A @ A_inv ≈ I.

```python
is_valid, max_error = inverter.verify_inverse(A, A_inv)
```

**Parameters:**
- `A` (ndarray): Original matrix
- `A_inv` (ndarray): Computed inverse
- `tolerance` (float): Maximum acceptable error

**Returns:**
- Tuple of (is_valid, max_error)

### InvertibilityChecker

Verifies matrix invertibility and analyzes numerical conditioning.

#### Constructor

```python
InvertibilityChecker(det_threshold=1e-10, cond_threshold=1e13)
```

**Parameters:**
- `det_threshold` (float): Minimum absolute determinant value
- `cond_threshold` (float): Maximum condition number for well-conditioned matrices

#### Methods

##### check(A)

Perform comprehensive invertibility check.

```python
result = checker.check(A)
```

**Parameters:**
- `A` (ndarray): Matrix to check

**Returns:**
- `InvertibilityResult` with fields:
  - `status`: InvertibilityStatus enum
  - `is_invertible`: Boolean
  - `condition_number`: κ(A)
  - `determinant`: det(A)
  - `message`: Descriptive message

##### compute_condition_number(A, norm='2')

Compute condition number using specified norm.

```python
cond = checker.compute_condition_number(A)
```

**Parameters:**
- `A` (ndarray): Input matrix
- `norm` (str): Norm type ('1', '2', 'fro', 'inf')

**Returns:**
- `float`: Condition number κ(A)

### StabilityAnalyzer

Analyzes numerical stability and error propagation.

#### Methods

##### analyze(A, A_inv, b=None)

Perform comprehensive stability analysis.

```python
report = analyzer.analyze(A, A_inv, b)
```

**Parameters:**
- `A` (ndarray): Original matrix
- `A_inv` (ndarray): Computed inverse
- `b` (ndarray, optional): Demand vector for residual analysis

**Returns:**
- `StabilityReport` with fields:
  - `condition_number`: κ(A)
  - `estimated_error_bound`: Theoretical error bound
  - `relative_residual`: Actual relative residual
  - `is_stable`: Boolean stability flag
  - `warnings`: List of warning messages
  - `recommendations`: List of recommendations

##### compare_methods(A, b)

Compare inversion-based solution with direct solving.

```python
comparison = analyzer.compare_methods(A, b)
```

**Parameters:**
- `A` (ndarray): Dependency matrix
- `b` (ndarray): Demand vector

**Returns:**
- Dictionary comparing different solution methods

## Exceptions

### AllocationException

Base exception for all allocation-related errors.

### InvalidMatrixException

Raised when matrix is invalid for inversion.

**Subclasses:**
- `NonSquareMatrixException`: Matrix is not square
- `SingularMatrixException`: Matrix is singular (non-invertible)
- `IllConditionedMatrixException`: Matrix is ill-conditioned

### DimensionMismatchException

Raised when dimensions don't match for operations.

### NumericalInstabilityException

Raised when numerical instability is detected.

### ComputationFailedException

Raised when computation fails unexpectedly.

## Data Classes

### InvertibilityResult

Result of invertibility check.

**Fields:**
- `status` (InvertibilityStatus): Status enumeration
- `is_invertible` (bool): Whether matrix is invertible
- `condition_number` (float): Condition number
- `determinant` (float or None): Determinant value
- `message` (str): Descriptive message

### StabilityReport

Comprehensive stability analysis report.

**Fields:**
- `condition_number` (float): κ(A)
- `estimated_error_bound` (float): Theoretical error bound
- `relative_residual` (float): Actual relative residual
- `is_stable` (bool): Stability flag
- `warnings` (list): Warning messages
- `recommendations` (list): Recommendations for improvement

## Enums

### InvertibilityStatus

Status enumeration for invertibility checks.

**Values:**
- `INVERTIBLE`: Matrix is invertible
- `NON_SQUARE`: Matrix is not square
- `SINGULAR`: Matrix is singular
- `ILL_CONDITIONED`: Matrix is ill-conditioned

## Usage Examples

### Basic Usage

```python
import numpy as np
from resource_allocation import AllocationSolver

# Define dependency matrix
A = np.array([
    [2.0, 1.0, 0.5],
    [1.5, 2.5, 1.0],
    [0.5, 1.0, 2.0]
])

# Initialize solver
solver = AllocationSolver(A)

# Compute allocation
demand = np.array([100, 150, 80])
allocation = solver.solve(demand)
print(f"Allocations: {allocation}")
```

### Batch Processing

```python
# Create multiple demand scenarios
scenarios = np.array([
    [100, 80, 120, 60],
    [150, 120, 180, 90],
    [80, 60, 100, 40]
])

# Compute all at once
allocations = solver.solve_batch(scenarios)
print(f"Batch allocations shape: {allocations.shape}")
```

### Stability Analysis

```python
from resource_allocation import StabilityAnalyzer

analyzer = StabilityAnalyzer()
report = analyzer.analyze(A, solver.get_inverse_matrix(), demand)

if not report.is_stable:
    print("Warnings:", report.warnings)
    print("Recommendations:", report.recommendations)
```

### Error Handling

```python
try:
    solver = AllocationSolver(singular_matrix)
except InvalidMatrixException as e:
    print(f"Cannot initialize solver: {e}")

try:
    allocation = solver.solve(wrong_size_vector)
except DimensionMismatchException as e:
    print(f"Dimension error: {e}")
```

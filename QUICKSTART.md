# Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/resource-allocation-project.git
   cd resource-allocation-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

## Running Examples

### Basic Usage Example

```bash
python examples/basic_usage.py
```

This will demonstrate:
- Matrix initialization
- Single allocation computation
- Batch processing
- Performance analysis

### Stability Analysis Example

```bash
python examples/stability_analysis.py
```

This demonstrates advanced stability analysis features.

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=resource_allocation --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Running Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

This will benchmark:
- Matrix inversion performance
- Allocation query performance
- Batch processing efficiency

## Your First Allocation

Create a file `my_first_allocation.py`:

```python
import numpy as np
from resource_allocation import AllocationSolver

# Define your system
# 3 resources (CPU, Memory, Bandwidth)
# 3 services (Service A, B, C)
A = np.array([
    [2.0, 1.0, 0.5],  # CPU cores per service
    [1.5, 2.5, 1.0],  # GB RAM per service
    [0.5, 1.0, 2.0]   # Gbps bandwidth per service
])

# Initialize solver
solver = AllocationSolver(A)

# Your current resource availability
demand = np.array([100, 150, 80])  # Total CPU, RAM, Bandwidth

# Compute optimal allocation
allocation = solver.solve(demand)

print("Optimal service allocations:")
print(f"  Service A: {allocation[0]:.2f} instances")
print(f"  Service B: {allocation[1]:.2f} instances")
print(f"  Service C: {allocation[2]:.2f} instances")
```

Run it:
```bash
python my_first_allocation.py
```

## Project Structure

```
resource-allocation-project/
├── resource_allocation/       # Main package
│   ├── __init__.py
│   ├── allocation_solver.py   # Main solver interface
│   ├── matrix_inverter.py     # Matrix inversion engine
│   ├── invertibility_checker.py
│   ├── stability_analyzer.py
│   └── exceptions.py
├── tests/                      # Test suite
│   ├── test_inverter.py
│   ├── test_solver.py
│   └── ...
├── examples/                   # Usage examples
│   ├── basic_usage.py
│   └── stability_analysis.py
├── benchmarks/                 # Performance benchmarks
│   └── run_benchmarks.py
├── docs/                       # Documentation
│   └── API_REFERENCE.md
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Project README
```

## Common Use Cases

### Cloud Resource Allocation

```python
# CPU, Memory, Network for microservices
A = np.array([
    [2.5, 1.0, 0.5, 3.0],  # CPU cores
    [4.0, 2.0, 1.5, 6.0],  # GB RAM
    [0.5, 0.2, 0.1, 1.0]   # Gbps network
])

solver = AllocationSolver(A)
demand = np.array([200, 300, 50])  # Available resources
allocation = solver.solve(demand)
```

### Manufacturing Resource Allocation

```python
# Power, Cooling, Materials for production lines
A = np.array([
    [100, 150, 80],   # kW power per line
    [50, 75, 40],     # Cooling units per line
    [20, 30, 15]      # Material units per line
])

solver = AllocationSolver(A)
demand = np.array([5000, 2500, 1000])  # Available capacity
allocation = solver.solve(demand)
```

### Network Bandwidth Allocation

```python
# Bandwidth allocation across QoS tiers
A = np.array([
    [10, 5, 2],   # Premium tier
    [5, 10, 3],   # Standard tier
    [2, 3, 10]    # Basic tier
])

solver = AllocationSolver(A)
demand = np.array([1000, 1500, 800])  # Mbps available
allocation = solver.solve(demand)
```

## Troubleshooting

### "Matrix is singular" error

Your dependency matrix has linearly dependent rows/columns. Check for:
- Duplicate services with identical resource requirements
- Redundant constraints
- Numerical precision issues

Solution: Review your dependency matrix construction.

### "Dimension mismatch" error

The demand vector size doesn't match the matrix dimensions.

Solution: Ensure `len(demand) == A.shape[0]`

### High condition number warning

Your matrix is ill-conditioned, which may affect accuracy.

Solution:
- Review matrix construction
- Consider regularization
- Use stability analysis to assess impact

## Next Steps

1. Read the [API Reference](docs/API_REFERENCE.md)
2. Explore [Examples](examples/)
3. Run [Benchmarks](benchmarks/)
4. Read [Contributing Guide](CONTRIBUTING.md)

## Getting Help

- Check the [API Reference](docs/API_REFERENCE.md)
- Review [Examples](examples/)
- Open an issue on GitHub
- Read the source code (it's well-documented!)

## License

MIT License - see [LICENSE](LICENSE) file for details.

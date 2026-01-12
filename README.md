# Real-Time Resource Allocation Using Inverse Matrices

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A production-ready software solution for real-time resource allocation in cloud-based distributed systems using inverse matrix computations.

## 🌟 Features

- **Fast Allocation**: Sub-millisecond resource allocation for typical workloads (50-100 services)
- **Numerical Stability**: Comprehensive stability testing with relative errors below 10⁻¹⁰
- **Modular Architecture**: Clean separation between mathematical operations and business logic
- **Batch Processing**: Efficient processing of multiple allocation requests simultaneously
- **Comprehensive Testing**: 90%+ test coverage with unit, integration, and numerical validation tests
- **Production Ready**: Complete API documentation and deployment guides

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from resource_allocation import AllocationSolver

# Define dependency matrix (3 resources, 3 services)
A = np.array([
    [2.0, 1.0, 0.5],  # CPU requirements
    [1.5, 2.5, 1.0],  # Memory requirements
    [0.5, 1.0, 2.0]   # Bandwidth requirements
])

# Initialize solver
solver = AllocationSolver(A)

# Compute allocation for demand vector
demand = np.array([100, 150, 80])
allocation = solver.solve(demand)

print(f"Service allocations: {allocation}")
```

## 📊 Performance

| Matrix Size | Inversion Time | Allocation Time | Throughput |
|-------------|----------------|-----------------|------------|
| 50×50       | < 100ms        | < 1ms           | > 1000 qps |
| 100×100     | < 500ms        | < 5ms           | > 200 qps  |
| 200×200     | < 3000ms       | < 20ms          | > 50 qps   |

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Usage Examples](examples/)
- [Performance Analysis](docs/performance.md)
- [Architecture Overview](docs/architecture.md)

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run benchmarks:

```bash
python benchmarks/inversion_benchmark.py
python benchmarks/allocation_benchmark.py
```

## 🏗️ Architecture

The system follows a layered architecture:

1. **API Layer**: RESTful interfaces for external integration
2. **Business Logic Layer**: Allocation solver and workflow coordination
3. **Mathematical Operations Layer**: Core matrix operations and algorithms
4. **Stability Analysis Layer**: Numerical diagnostics and error detection
5. **Utilities Layer**: Logging, configuration, and helper functions

## 📦 Project Structure

```
resource-allocation-project/
├── resource_allocation/       # Main package
│   ├── __init__.py
│   ├── matrix_inverter.py    # Matrix inversion engine
│   ├── invertibility_checker.py
│   ├── allocation_solver.py  # Main solver interface
│   ├── stability_analyzer.py
│   └── exceptions.py
├── tests/                     # Test suite
│   ├── test_inverter.py
│   ├── test_checker.py
│   ├── test_solver.py
│   └── test_stability.py
├── benchmarks/               # Performance benchmarks
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── requirements.txt
```

## 🎯 Use Cases

- **Cloud Resource Orchestration**: Dynamic allocation of CPU, memory, and bandwidth
- **Manufacturing Process Control**: Power, cooling, and material allocation
- **Network Traffic Engineering**: Bandwidth allocation across network links

## 🔬 Mathematical Background

The resource allocation problem is formulated as:

```
Ax = b
```

Where:
- **A** ∈ ℝⁿˣⁿ is the resource-to-service dependency matrix
- **x** ∈ ℝⁿ is the vector of resource allocations
- **b** ∈ ℝⁿ is the vector of observed demands

By precomputing A⁻¹, allocations can be computed as:

```
x = A⁻¹b
```

This trades an O(n³) inversion cost for O(n²) query costs on subsequent allocations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Syuzanna Ghazaryan** - *Initial work* - French University in Armenia

## 🙏 Acknowledgments

- Supervisor: Yeghisabet Alaverdyan
- French University in Armenia - Faculty of Computer Science and Applied Mathematics

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔮 Future Enhancements

- Sparse matrix support for large-scale systems
- Incremental matrix updates using Sherman-Morrison formula
- GPU acceleration for massive deployments
- Machine learning integration for demand prediction
- Automatic regularization for ill-conditioned matrices

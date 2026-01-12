# Contributing to Resource Allocation System

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/resource-allocation-project.git
   cd resource-allocation-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

## Running Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=resource_allocation --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Code Style

This project follows PEP 8 style guidelines. We use:
- `black` for code formatting
- `pylint` for linting
- `mypy` for type checking

Format your code:
```bash
black resource_allocation/ tests/ examples/
```

Check code quality:
```bash
pylint resource_allocation/
mypy resource_allocation/ --ignore-missing-imports
```

## Contribution Workflow

1. **Fork the repository** on GitHub

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Pull Request Guidelines

- **Title**: Clear, concise description of changes
- **Description**: Explain what changes were made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Code Style**: Follow project style guidelines

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version
- NumPy/SciPy versions
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

When proposing features:
- Describe the use case
- Explain expected behavior
- Consider backwards compatibility
- Discuss implementation approach

### Code Contributions

Focus areas for contributions:
- **Performance**: Optimization of matrix operations
- **Sparse Matrices**: Support for sparse dependency matrices
- **GPU Acceleration**: CUDA/OpenCL implementations
- **Documentation**: Examples, tutorials, API docs
- **Testing**: Additional test cases, edge cases

## Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Use meaningful test names
- Cover edge cases and error conditions
- Aim for >90% coverage

### Integration Tests
- Test component interactions
- Verify end-to-end workflows
- Test with realistic data

### Performance Tests
- Benchmark critical operations
- Monitor for performance regressions
- Document expected performance

## Documentation

### Code Documentation
- Use docstrings for all public functions/classes
- Follow NumPy docstring format
- Include examples in docstrings

### Example Docstring
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function.
    
    Longer description explaining functionality,
    algorithms, and considerations.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

## Questions?

If you have questions:
- Open an issue on GitHub
- Check existing issues and documentation
- Review the code examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Resource Allocation System!

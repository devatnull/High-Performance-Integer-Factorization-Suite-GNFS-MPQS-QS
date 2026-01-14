# Contributing to Integer Factorization Suite

Thanks for your interest in contributing! This project implements various integer factorization algorithms from simple trial division to the General Number Field Sieve.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/devatnull/High-Performance-Integer-Factorization-Suite-GNFS-MPQS-QS.git
cd High-Performance-Integer-Factorization-Suite-GNFS-MPQS-QS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy sympy pytest

# Optional: faster arithmetic
pip install gmpy2 numba

# Run tests
PYTHONPATH=. pytest tests/ -v
```

## Project Structure

```
factorization/
├── __init__.py      # Main API, auto-selection
├── __main__.py      # CLI: python -m factorization
├── tui.py           # Interactive TUI
├── utils.py         # Math utilities (GCD, primality, etc.)
├── trial.py         # Trial division
├── fermat.py        # Fermat's method
├── pollard_rho.py   # Pollard Rho (Brent's improvement)
├── squfof.py        # Shanks' SQUFOF
├── pollard_pm1.py   # Pollard p-1
├── williams_pp1.py  # Williams p+1
├── ecm.py           # Elliptic Curve Method
├── qs.py            # Quadratic Sieve
├── mpqs.py          # Multiple Polynomial QS (SIQS)
├── gnfs.py          # GNFS (basic)
├── gnfs_real.py     # GNFS (production structure)
├── linear_algebra.py # GF(2) matrix operations
├── simd.py          # Numba JIT acceleration
└── parallel.py      # Multiprocessing
```

## Code Style

- **Type hints** everywhere (`List`, `Tuple`, `Optional`, etc.)
- **Docstrings** explain the math, not just the API
- Each algorithm is **self-contained** with paper references
- Follow existing patterns in the codebase

Example:
```python
def my_algorithm(n: int, max_iterations: int = 1000) -> Optional[int]:
    """
    Brief description of the algorithm.
    
    Mathematical background explaining WHY it works.
    
    Complexity: O(n^1/4) expected
    
    References:
        - Author, "Paper Title" (Year)
    
    Args:
        n: Number to factor
        max_iterations: Maximum iterations before giving up
    
    Returns:
        A non-trivial factor of n, or None if not found
    
    Example:
        >>> my_algorithm(143)
        11
    """
```

## Areas for Contribution

### High Impact
- **GNFS algebraic square root**: Implement proper Schirokauer maps and quadratic characters
- **GPU acceleration**: CUDA/OpenCL sieving
- **Block Wiedemann**: For large sparse matrices

### Medium Impact
- **Large prime variation** for QS/MPQS
- **Better polynomial selection** for GNFS (Kleinjung's full method)
- **Distributed sieving** support

### Good First Issues
- Add more test cases
- Improve documentation
- Optimize hot loops with Numba
- Add progress callbacks

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test class
PYTHONPATH=. pytest tests/test_factorization.py::TestECM -v

# Skip slow tests
PYTHONPATH=. pytest tests/ -v -m "not slow"
```

When adding new features:
1. Add tests in `tests/test_factorization.py`
2. Ensure all existing tests pass
3. Test with various input sizes

## Benchmarking

```bash
# Quick benchmark
python benchmark.py --quick

# Use the TUI
python -m factorization.tui
# Then press 'b' for benchmark
```

## Pull Request Process

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Ensure tests pass: `PYTHONPATH=. pytest tests/`
5. Commit with clear message: `git commit -m "Add: description"`
6. Push and open PR

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the math/algorithms

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

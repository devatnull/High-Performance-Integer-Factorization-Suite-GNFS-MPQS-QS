# Integer Factorization Suite

A comprehensive Python library implementing state-of-the-art integer factorization algorithms, from simple trial division to the General Number Field Sieve.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-34%20passing-brightgreen.svg" alt="Tests">
</p>

## Features

- **10 factorization algorithms** with automatic selection based on input size
- **Interactive TUI** for easy testing and benchmarking
- **~6,000 lines** of documented, type-hinted Python
- **34 unit tests** covering all algorithms
- **CLI interface** for quick factorization
- **Optional acceleration** via gmpy2 and Numba JIT

## Algorithms

| Algorithm | Complexity | Best For |
|-----------|------------|----------|
| Trial Division | O(√n) | n < 10¹² |
| Fermat's Method | O(√n) | Factors close together (p ≈ q) |
| Pollard's Rho | O(n¹/⁴) | 10¹⁰ < n < 10²⁵ |
| SQUFOF | O(n¹/⁴) | Same range, constant memory |
| Pollard p-1 | O(B log B) | Smooth (p-1) factors |
| Williams p+1 | O(B log B) | Smooth (p+1) factors |
| ECM | O(exp(√(ln p ln ln p))) | Medium factors in large n |
| Quadratic Sieve | L[1/2, 1] | 10²⁵ < n < 10⁵⁰ |
| MPQS (SIQS) | L[1/2, 1] | 10²⁵ < n < 10¹⁰⁰ |
| GNFS | L[1/3, 1.923] | n > 10¹⁰⁰ |

## Installation

```bash
# Clone and install
git clone https://github.com/devatnull/High-Performance-Integer-Factorization-Suite-GNFS-MPQS-QS.git
cd High-Performance-Integer-Factorization-Suite-GNFS-MPQS-QS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy sympy pytest

# Optional: faster arithmetic
pip install gmpy2 numba
```

## Quick Start

### Interactive TUI (Recommended)

```bash
python -m factorization.tui
```

This launches an interactive interface where you can:
- Factor numbers with auto-selected or specific algorithms
- Run benchmarks
- Estimate GNFS resources for large numbers
- Check primality

### Python API

```python
from factorization import factorize, factorize_full, is_prime

# Auto-selects best algorithm
p, q = factorize(1000000007 * 1000000009)
print(f"{p} × {q}")  # 1000000007 × 1000000009

# Full prime factorization
factors = factorize_full(2**10 * 3**5 * 17)
print(factors)  # {2: 10, 3: 5, 17: 1}

# Use specific algorithms
from factorization import pollard_rho, ecm, mpqs_factor

factor = pollard_rho(n)
factor = ecm(n, B1=50000, max_curves=100)
p, q = mpqs_factor(n, time_limit=60, verbose=True)
```

### Command Line

```bash
# Basic usage
python -m factorization 1234567890123

# Specify algorithm
python -m factorization 99999999999999997 --algorithm ecm

# Full factorization with verbose output
python -m factorization 12345678901234567890 --full --verbose

# Show available algorithms
python -m factorization --info
```

## Algorithm Details

### Automatic Selection

The `factorize()` function automatically selects the best algorithm:

- **< 20 digits**: Trial division + Pollard Rho
- **20-50 digits**: ECM with MPQS fallback  
- **50-80 digits**: MPQS
- **> 80 digits**: GNFS (experimental)

### ECM (Elliptic Curve Method)

Best for finding medium-sized factors (< 60 digits) regardless of n's size:

```python
from factorization import ecm

# B1 controls smoothness bound - larger finds bigger factors
factor = ecm(n, B1=50000, B2=5000000, max_curves=200)
```

Recommended B1 values:
- 20-digit factors: B1 = 11,000
- 30-digit factors: B1 = 250,000
- 40-digit factors: B1 = 3,000,000

### MPQS (Self-Initializing Quadratic Sieve)

For semiprimes in the 25-60 digit range:

```python
from factorization import mpqs_factor

p, q = mpqs_factor(n, time_limit=300, verbose=True)          # auto-select workers
p, q = mpqs_factor(n, time_limit=300, verbose=True, num_workers=8)
```

MPQS now auto-enables multicore sieving for MPQS-sized inputs and falls back to
single-process execution when process overhead would dominate.

### GNFS (General Number Field Sieve)

The asymptotically fastest classical algorithm. Our implementation includes:

- Base-m and Kleinjung polynomial selection
- Line sieving and lattice sieving
- Gaussian elimination over GF(2)
- ECM fallback for square root failures

```python
from factorization import gnfs_factor_real, estimate_gnfs_runtime

# See what resources would be needed
est = estimate_gnfs_runtime(2**512)
print(f"RSA-512 would need ~{est['core_years']:.0f} core-years")

# Attempt factorization (works up to ~60 digits reliably)
p, q = gnfs_factor_real(n, time_limit=600, verbose=True)
```

## GNFS Resource Estimates

| Key Size | Digits | Core-Years | Status |
|----------|--------|------------|--------|
| RSA-512 | 155 | ~0.3 | Broken easily |
| RSA-768 | 232 | ~2,000 | Factored 2009 |
| RSA-1024 | 309 | ~10⁶ | Theoretically reachable |
| RSA-2048 | 617 | ~10¹⁵ | Computationally infeasible |
| RSA-4096 | 1234 | ~10²⁷ | Impossible |

## Benchmarks

Run the benchmark suite:

```bash
python benchmark.py --quick
```

Example output on Apple M1:
```
Testing 12-digit semiprimes:
  Pollard Rho... 0.001s avg, 100% success
  ECM...         0.002s avg, 100% success
  MPQS...        0.210s avg, 100% success

Testing 16-digit semiprimes:
  Pollard Rho... 0.001s avg, 100% success
  ECM...         0.003s avg, 100% success
  MPQS...        0.684s avg, 100% success
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Project Structure

```
factorization/
├── __init__.py      # Main API, auto-selection logic
├── __main__.py      # CLI interface
├── utils.py         # GCD, primality testing, Tonelli-Shanks
├── trial.py         # Trial division
├── fermat.py        # Fermat's factorization
├── pollard_rho.py   # Pollard Rho with Brent's improvement
├── squfof.py        # Shanks' Square Forms Factorization
├── pollard_pm1.py   # Pollard p-1 method
├── williams_pp1.py  # Williams p+1 method
├── ecm.py           # Elliptic Curve Method (Montgomery curves)
├── qs.py            # Quadratic Sieve
├── mpqs.py          # Self-Initializing QS (SIQS)
├── gnfs.py          # GNFS (basic implementation)
├── gnfs_real.py     # GNFS (production structure)
├── linear_algebra.py # GF(2) Gaussian elimination
├── simd.py          # Numba JIT acceleration
└── parallel.py      # Multiprocessing support
```

## Mathematical References

- Lenstra, H.W. "Factoring integers with elliptic curves" (1987)
- Pomerance, C. "The Quadratic Sieve Factoring Algorithm" (1984)
- Silverman, R. "The Multiple Polynomial Quadratic Sieve" (1987)
- Pollard, J.M. "A Monte Carlo method for factorization" (1975)
- Brent, R.P. "An improved Monte Carlo factorization algorithm" (1980)
- Shanks, D. "Analysis and Improvement of the Continued Fraction Method" (1975)
- Lenstra, Lenstra, Manasse, Pollard "The Number Field Sieve" (1993)

## Limitations

- **GNFS square root**: The algebraic square root computation requires handling non-principal ideals in the class group of the number field. Our implementation attempts direct computation but falls back to ECM when the ideal class structure prevents integer solutions. Full GNFS (CADO-NFS, msieve) uses Schirokauer maps and quadratic characters to handle this - roughly 10,000+ additional lines of code.
- No GPU acceleration (yet)
- Single-machine only; no distributed sieving
- Numbers > 80 digits may timeout or require parameter tuning

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas that could use work:
- **GNFS square root**: Schirokauer maps, quadratic characters
- **Block Wiedemann**: For large sparse matrices  
- **GPU sieving**: CUDA/OpenCL acceleration
- **Distributed sieving**: Multi-machine support

## License

MIT License - see [LICENSE](LICENSE) file.

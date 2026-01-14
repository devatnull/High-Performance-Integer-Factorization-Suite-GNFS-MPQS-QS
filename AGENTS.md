# AGENTS.md

## Commands
```bash
python3 -m venv .venv && source .venv/bin/activate  # Setup venv
pip install numpy scipy sympy pytest                 # Dependencies
pytest tests/ -v                                     # Run test suite
python benchmark.py --quick                          # Quick benchmark
python -m factorization 1234567                      # CLI usage
python -m factorization --info                       # Show available algorithms
```

## Architecture
Modular package in `factorization/`:
- `utils.py` - GCD, primality, Tonelli-Shanks, factor base utilities  
- `trial.py` - Trial division O(√n)
- `fermat.py` - Fermat's difference of squares (good for p≈q)
- `pollard_rho.py` - Brent's cycle detection O(n^1/4)
- `squfof.py` - Shanks' Square Forms O(n^1/4)
- `pollard_pm1.py` - Two-stage p-1 for smooth (p-1)
- `williams_pp1.py` - p+1 method using Lucas sequences
- `ecm.py` - Montgomery curves, Lenstra's ECM
- `qs.py` - Quadratic Sieve with log sieving
- `mpqs.py` - Self-Initializing QS (SIQS)
- `gnfs.py` - General Number Field Sieve (experimental)
- `linear_algebra.py` - GF(2) Gaussian elimination
- `simd.py` - Numba JIT-compiled hot loops
- `parallel.py` - Multiprocessing support
- `__main__.py` - CLI interface

Auto-selection in `__init__.py`: trial → Pollard Rho → ECM → MPQS based on size.

## Code Style
- Type hints everywhere (List, Tuple, Optional, etc.)
- Docstrings explain the math, not just the API
- `gmpy2` optional for 10-100x speedup on big integers
- Each algorithm self-contained with clear theory references

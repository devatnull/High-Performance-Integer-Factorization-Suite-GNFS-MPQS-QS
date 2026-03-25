"""
SIMD-style optimizations using Numba JIT compilation.

This module provides JIT-compiled versions of performance-critical
operations used in sieving algorithms. Numba compiles Python to 
machine code, giving C-like performance.

The key insight: most time in QS/MPQS is spent in:
1. Sieve array updates (subtracting log values)
2. Smooth number testing (trial division)
3. GCD computations

By JIT-compiling these inner loops, we get 10-50x speedup.

Usage:
    from factorization.simd import fast_sieve, fast_is_smooth
    
    # These are drop-in replacements for the pure Python versions
    fast_sieve(sieve_array, prime, log_prime, start1, start2)
    is_smooth = fast_is_smooth(n, factor_base_array)
"""

import os
import numpy as np
from typing import List, Tuple, Optional

# Try to import numba - graceful fallback if not available
try:
    if os.environ.get("FACTOR_DISABLE_NUMBA") == "1":
        raise ImportError("Numba disabled by environment")
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def fast_gcd(a: int, b: int) -> int:
    """JIT-compiled Euclidean GCD."""
    while b:
        a, b = b, a % b
    return a


@jit(nopython=True, cache=True)
def fast_powmod(base: int, exp: int, mod: int) -> int:
    """JIT-compiled modular exponentiation (binary method)."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


@jit(nopython=True, cache=True)
def fast_is_smooth(n: int, factor_base: np.ndarray) -> bool:
    """
    JIT-compiled smoothness test.
    
    Returns True if n factors completely over the factor base.
    Much faster than Python version due to tight loop compilation.
    """
    for i in range(len(factor_base)):
        p = factor_base[i]
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    return n == 1 or n == -1


@jit(nopython=True, cache=True)
def fast_factor_over_base(n: int, factor_base: np.ndarray, 
                          exponents: np.ndarray) -> bool:
    """
    JIT-compiled factorization over base.
    
    Fills exponents array with prime exponents, returns True if smooth.
    """
    for i in range(len(factor_base)):
        p = factor_base[i]
        exp = 0
        while n % p == 0:
            n //= p
            exp += 1
        exponents[i] = exp
    return n == 1 or n == -1


@jit(nopython=True, cache=True, parallel=True)
def fast_sieve_interval(sieve_log: np.ndarray, 
                        factor_base: np.ndarray,
                        log_primes: np.ndarray,
                        starts1: np.ndarray,
                        starts2: np.ndarray) -> None:
    """
    JIT-compiled logarithmic sieving.
    
    For each prime p in factor base:
    - Subtract log(p) from positions divisible by p
    - Handle both roots (start1, start2)
    
    This is the innermost loop of QS/MPQS - optimizing it is critical.
    Uses parallel execution when possible.
    """
    sieve_size = len(sieve_log)
    
    for j in range(len(factor_base)):
        p = factor_base[j]
        log_p = log_primes[j]
        start1 = starts1[j]
        start2 = starts2[j]
        
        # Sieve from start1
        if start1 >= 0:
            i = start1
            while i < sieve_size:
                sieve_log[i] -= log_p
                i += p
        
        # Sieve from start2 (if different)
        if start2 >= 0 and start2 != start1:
            i = start2
            while i < sieve_size:
                sieve_log[i] -= log_p
                i += p


@jit(nopython=True, cache=True)
def fast_init_sieve_log(sieve_log: np.ndarray, 
                        sieve_start: int, 
                        sqrt_n: int,
                        n: int) -> None:
    """
    Initialize sieve array with log|Q(x)| values.
    
    Q(x) = (sieve_start + i + sqrt_n)² - n
    """
    for i in range(len(sieve_log)):
        x = sieve_start + i + sqrt_n
        qx = x * x - n
        if qx > 0:
            # Fast log approximation using bit length
            sieve_log[i] = qx.bit_length() * 0.693147  # ln(2) ≈ 0.693
        elif qx < 0:
            sieve_log[i] = (-qx).bit_length() * 0.693147
        else:
            sieve_log[i] = 0.0


@jit(nopython=True, cache=True)
def fast_find_smooth_candidates(sieve_log: np.ndarray, 
                                threshold: float) -> np.ndarray:
    """
    Find indices where sieve_log < threshold (potential smooth numbers).
    
    Returns array of indices.
    """
    count = 0
    for i in range(len(sieve_log)):
        if sieve_log[i] < threshold:
            count += 1
    
    result = np.empty(count, dtype=np.int64)
    j = 0
    for i in range(len(sieve_log)):
        if sieve_log[i] < threshold:
            result[j] = i
            j += 1
    
    return result


@jit(nopython=True, cache=True)
def fast_matrix_mult_gf2(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication over GF(2).
    
    Returns A @ v mod 2.
    """
    m, k = A.shape
    result = np.zeros(m, dtype=np.int8)
    
    for i in range(m):
        s = 0
        for j in range(k):
            s ^= A[i, j] & v[j]
        result[i] = s
    
    return result


@jit(nopython=True, cache=True)
def fast_gaussian_step(matrix: np.ndarray, row: int, col: int) -> int:
    """
    One step of Gaussian elimination over GF(2).
    
    Find pivot in column `col` starting from `row`, swap if needed,
    eliminate column in other rows.
    
    Returns the new row index after this pivot, or -1 if no pivot found.
    """
    m, k = matrix.shape
    
    # Find pivot
    pivot_row = -1
    for r in range(row, m):
        if matrix[r, col] == 1:
            pivot_row = r
            break
    
    if pivot_row == -1:
        return -1
    
    # Swap rows
    if pivot_row != row:
        for j in range(k):
            matrix[row, j], matrix[pivot_row, j] = matrix[pivot_row, j], matrix[row, j]
    
    # Eliminate
    for r in range(m):
        if r != row and matrix[r, col] == 1:
            for j in range(k):
                matrix[r, j] ^= matrix[row, j]
    
    return row + 1


@jit(nopython=True, cache=True)
def fast_tonelli_shanks(n: int, p: int) -> int:
    """
    JIT-compiled Tonelli-Shanks square root algorithm.
    
    Returns r such that r² ≡ n (mod p), or -1 if no root exists.
    """
    # Check if n is a quadratic residue
    if fast_powmod(n, (p - 1) // 2, p) != 1:
        return -1
    
    # Factor out powers of 2: p - 1 = Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # Simple case: p ≡ 3 (mod 4)
    if S == 1:
        return fast_powmod(n, (p + 1) // 4, p)
    
    # Find quadratic non-residue z
    z = 2
    while fast_powmod(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    M = S
    c = fast_powmod(z, Q, p)
    t = fast_powmod(n, Q, p)
    R = fast_powmod(n, (Q + 1) // 2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Update
        b = c
        for _ in range(M - i - 1):
            b = (b * b) % p
        
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R


# Convenience function to check if Numba is available
def numba_available() -> bool:
    """Check if Numba JIT compilation is available."""
    return HAS_NUMBA


# Wrapper functions that convert Python lists to numpy arrays
def sieve_with_numba(sieve_log: np.ndarray, factor_base: List[int],
                     starts: List[Tuple[int, int]], n: int, sqrt_n: int,
                     sieve_start: int) -> None:
    """
    Wrapper for fast sieving that handles type conversion.
    """
    if not HAS_NUMBA:
        # Fallback to pure Python
        import math
        for j, p in enumerate(factor_base):
            log_p = math.log(p)
            start1, start2 = starts[j]
            if start1 >= 0:
                for i in range(start1, len(sieve_log), p):
                    sieve_log[i] -= log_p
            if start2 >= 0 and start2 != start1:
                for i in range(start2, len(sieve_log), p):
                    sieve_log[i] -= log_p
        return
    
    # Convert to numpy arrays
    fb_array = np.array(factor_base, dtype=np.int64)
    log_primes = np.log(fb_array.astype(np.float64))
    starts1 = np.array([s[0] for s in starts], dtype=np.int64)
    starts2 = np.array([s[1] for s in starts], dtype=np.int64)
    
    fast_sieve_interval(sieve_log, fb_array, log_primes, starts1, starts2)


def is_smooth_numba(n: int, factor_base: List[int]) -> bool:
    """Wrapper for fast smoothness test."""
    if not HAS_NUMBA:
        # Fallback
        for p in factor_base:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        return abs(n) == 1
    
    return fast_is_smooth(n, np.array(factor_base, dtype=np.int64))


def factor_over_base_numba(n: int, factor_base: List[int]) -> Optional[List[int]]:
    """
    Wrapper for fast factorization over the base.

    Returns the exponent vector if ``n`` is smooth over ``factor_base``,
    otherwise ``None``.
    """
    if not HAS_NUMBA:
        from .utils import factor_over_base

        return factor_over_base(n, factor_base)

    sign = -1 if n < 0 else 1
    value = abs(n)
    factor_base_array = np.array(factor_base, dtype=np.int64)
    exponents = np.zeros(len(factor_base), dtype=np.int64)
    is_smooth = fast_factor_over_base(value, factor_base_array, exponents)
    if not is_smooth:
        return None

    if sign < 0 and exponents.size > 0 and factor_base[0] == -1:
        exponents[0] += 1

    return exponents.tolist()

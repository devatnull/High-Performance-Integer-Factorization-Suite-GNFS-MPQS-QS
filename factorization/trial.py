"""
Trial Division Factorization.

The simplest factorization algorithm: test divisibility by each prime
up to sqrt(n). Optimal for small factors and numbers up to ~10^12.

Time complexity: O(sqrt(n) / ln(sqrt(n))) assuming prime generation
"""

from typing import Tuple, List, Optional
from .utils import isqrt, is_prime, generate_primes


def trial_division(n: int, limit: Optional[int] = None) -> Tuple[int, int]:
    """
    Factor n using trial division.
    
    Tests divisibility by 2 and odd numbers up to min(sqrt(n), limit).
    
    Args:
        n: Number to factor
        limit: Maximum divisor to try (default: sqrt(n))
    
    Returns:
        (factor, cofactor) where factor * cofactor = n
        Returns (n, 1) if no factor found
    
    Example:
        >>> trial_division(15)
        (3, 5)
        >>> trial_division(1000003 * 1000033)  # Too large for trial division
        (1000036000099, 1)
    """
    if n <= 1:
        return (n, 1)
    
    if n % 2 == 0:
        return (2, n // 2)
    
    sqrt_n = isqrt(n)
    max_divisor = min(sqrt_n, limit) if limit else sqrt_n
    
    # Trial division by odd numbers
    d = 3
    while d <= max_divisor:
        if n % d == 0:
            return (d, n // d)
        d += 2
    
    return (n, 1)


def trial_division_full(n: int, limit: Optional[int] = None) -> dict:
    """
    Completely factor n using trial division.
    
    Returns dict mapping prime factors to their exponents.
    Remaining unfactored part (if any) is stored under key 'remainder'.
    
    Example:
        >>> trial_division_full(360)
        {2: 3, 3: 2, 5: 1}
        >>> trial_division_full(1000003 * 7)
        {7: 1, 'remainder': 1000003}
    """
    if n <= 1:
        return {}
    
    factors = {}
    sqrt_n = isqrt(n)
    max_divisor = min(sqrt_n, limit) if limit else sqrt_n
    
    # Factor out 2s
    exp = 0
    while n % 2 == 0:
        n //= 2
        exp += 1
    if exp > 0:
        factors[2] = exp
    
    # Factor out odd primes
    d = 3
    while d <= min(isqrt(n), max_divisor):
        exp = 0
        while n % d == 0:
            n //= d
            exp += 1
        if exp > 0:
            factors[d] = exp
        d += 2
    
    if n > 1:
        if is_prime(n):
            factors[n] = 1
        else:
            factors['remainder'] = n
    
    return factors


def trial_division_with_sieve(n: int, B: int = 100000) -> Tuple[int, int]:
    """
    Trial division using a precomputed prime sieve.
    
    More efficient than basic trial division for numbers where we expect
    small factors, as we only test actual primes.
    
    Args:
        n: Number to factor
        B: Smoothness bound - generate primes up to B
    
    Returns:
        (factor, cofactor) or (n, 1) if no small factor found
    """
    if n <= 1:
        return (n, 1)
    
    primes = generate_primes(min(B, isqrt(n) + 1))
    
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            return (p, n // p)
    
    return (n, 1)

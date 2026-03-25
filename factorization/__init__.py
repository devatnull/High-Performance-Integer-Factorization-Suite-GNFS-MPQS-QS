"""
High-Performance Integer Factorization Suite.

A comprehensive collection of integer factorization algorithms, from
simple trial division to advanced sieving methods. Each algorithm is
implemented with clear documentation of the underlying mathematics.

Algorithms by complexity and use case:

    Trial Division: O(√n)
        Best for: n < 10^12, finding small factors
        
    Pollard's Rho: O(n^1/4) expected
        Best for: 10^10 < n < 10^25, general purpose
        
    Pollard p-1: O(B log B log n)
        Best for: Primes p where (p-1) is smooth
        
    ECM (Elliptic Curve Method): O(exp(√(2 ln p ln ln p)))
        Best for: Finding medium factors (< 60 digits) in large n
        
    Quadratic Sieve: L_n[1/2, 1]
        Best for: 10^25 < n < 10^100, general purpose
        
    MPQS: L_n[1/2, 1] with better constants
        Best for: Same range as QS, faster in practice
        
    GNFS: L_n[1/3, (64/9)^(1/3)]  
        Best for: n > 10^100 (research/cryptanalysis)

Example usage:
    >>> from factorization import factorize
    >>> factorize(1000003 * 1000033)
    (1000003, 1000033)
    
    >>> from factorization import factorize_full
    >>> factorize_full(2**10 * 3**5 * 17)
    {2: 10, 3: 5, 17: 1}
"""

from .utils import (
    gcd, isqrt, is_prime, is_perfect_power,
    legendre_symbol, jacobi_symbol, tonelli_shanks,
    mod_inverse, powmod, generate_primes, gmpy2_available
)

from .trial import trial_division, trial_division_full
from .pollard_rho import pollard_rho, pollard_rho_factorize
from .pollard_pm1 import pollard_pm1
from .ecm import ecm, ecm_factorize
from .qs import qs_factor
from .mpqs import mpqs_factor
from .gnfs import gnfs_factor
from .gnfs_real import gnfs_factor_real, estimate_gnfs_runtime
from .williams_pp1 import williams_pp1
from .fermat import fermat_factor, fermat_factor_optimized
from .squfof import squfof, squfof_factor
from .simd import numba_available


def factorize(n: int, time_limit: float = 300, verbose: bool = False):
    """
    Automatically factor n using the best algorithm for its size.
    
    Algorithm selection based on digit count:
        < 12 digits: Trial division + Pollard Rho
        12-20 digits: Pollard Rho + ECM fallback
        20-50 digits: ECM + MPQS fallback
        50-100 digits: MPQS
        > 100 digits: Would use GNFS (not fully implemented)
    
    Args:
        n: Positive integer to factor
        time_limit: Maximum seconds for factorization attempt
        verbose: Print progress information
    
    Returns:
        (p, q) where p * q = n and p <= q
        Returns (n, 1) if n is prime or factorization fails
    
    Example:
        >>> factorize(15)
        (3, 5)
        >>> factorize(1000000007 * 1000000009)
        (1000000007, 1000000009)
    """
    if n <= 1:
        return (n, 1)
    
    # Check primality first
    if is_prime(n):
        return (n, 1)
    
    # Check for perfect power
    power = is_perfect_power(n)
    if power:
        base, exp = power
        sub_factor = factorize(base, time_limit / 2, verbose)
        return (sub_factor[0], n // sub_factor[0])
    
    digits = len(str(n))
    
    # Trial division for small factors (always try)
    limit = min(100000, int(n ** 0.5) + 1)
    p, q = trial_division(n, limit)
    if p != n:
        if verbose:
            print(f"Trial division found factor: {p}")
        return (min(p, q), max(p, q))
    
    # Small numbers: Pollard Rho is usually enough
    if digits <= 20:
        if verbose:
            print("Using Pollard Rho...")
        factor = pollard_rho(n)
        if factor:
            q = n // factor
            return (min(factor, q), max(factor, q))
    
    # Medium numbers: Try ECM for a bit, then MPQS
    if digits <= 50:
        if verbose:
            print("Using ECM...")
        # Scale ECM parameters to number size
        B1 = 10000 * (digits // 10)
        factor = ecm(n, B1=B1, B2=B1*100, max_curves=50)
        if factor:
            q = n // factor
            return (min(factor, q), max(factor, q))
    
    # Large numbers: MPQS for < 100 digits, GNFS for larger
    if digits <= 80:
        if verbose:
            print("Using MPQS...")
        result = mpqs_factor(n, time_limit * 0.7, verbose)
        if result[0] != n:
            return result
    
    # Very large numbers or MPQS failed: use GNFS
    if digits > 50:
        if verbose:
            print("Using GNFS...")
        return gnfs_factor(n, time_limit, verbose)
    
    return mpqs_factor(n, time_limit, verbose)


def factorize_full(n: int, time_limit: float = 300, verbose: bool = False) -> dict:
    """
    Completely factor n into prime powers.
    
    Returns:
        Dict mapping prime factors to their exponents
    
    Example:
        >>> factorize_full(360)
        {2: 3, 3: 2, 5: 1}
        >>> factorize_full(1000000007)  # Prime
        {1000000007: 1}
    """
    if n <= 1:
        return {}
    
    factors = {}
    remaining = n
    
    while remaining > 1:
        if is_prime(remaining):
            factors[remaining] = factors.get(remaining, 0) + 1
            break
        
        p, q = factorize(remaining, time_limit, verbose)
        
        if p == remaining:
            # Factorization failed
            factors[remaining] = factors.get(remaining, 0) + 1
            break
        
        # Count exponent of p
        exp = 0
        while remaining % p == 0:
            remaining //= p
            exp += 1
        
        if exp > 0:
            factors[p] = factors.get(p, 0) + exp
    
    return dict(sorted(factors.items()))


__all__ = [
    # Main API
    'factorize', 'factorize_full',
    
    # Individual algorithms
    'trial_division', 'trial_division_full',
    'fermat_factor', 'fermat_factor_optimized',
    'pollard_rho', 'pollard_rho_factorize',
    'squfof', 'squfof_factor',
    'pollard_pm1', 'williams_pp1',
    'ecm', 'ecm_factorize',
    'qs_factor', 'mpqs_factor', 'gnfs_factor', 
    'gnfs_factor_real', 'estimate_gnfs_runtime',
    
    # Utilities
    'gcd', 'isqrt', 'is_prime', 'is_perfect_power',
    'legendre_symbol', 'jacobi_symbol', 'tonelli_shanks',
    'mod_inverse', 'powmod', 'generate_primes',
    'numba_available', 'gmpy2_available'
]

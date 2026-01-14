"""
Pollard's Rho Algorithm for Integer Factorization.

A probabilistic algorithm based on the birthday paradox and cycle detection.
Expected running time: O(n^(1/4)) for finding a factor.

The key insight: if we have a sequence x_i = f(x_{i-1}) mod n, and p|n,
then the sequence mod p will cycle faster than mod n. Detecting this
cycle reveals the factor p.

References:
    - Pollard, J. M. "A Monte Carlo method for factorization" (1975)
    - Brent, R. P. "An improved Monte Carlo factorization algorithm" (1980)
"""

import random
from typing import Optional, Callable
from .utils import gcd, is_prime, isqrt


def pollard_rho_basic(n: int, max_iterations: int = 1000000) -> Optional[int]:
    """
    Basic Pollard's Rho with Floyd's cycle detection.
    
    Uses the tortoise-and-hare algorithm: two pointers moving at different
    speeds through the sequence. When they meet, gcd reveals a factor.
    
    The iteration function f(x) = x² + c (mod n) generates a pseudo-random
    sequence that cycles due to the pigeonhole principle.
    
    Args:
        n: Number to factor (should be composite and not a prime power)
        max_iterations: Maximum iterations before giving up
    
    Returns:
        A non-trivial factor of n, or None if not found
    
    Example:
        >>> pollard_rho_basic(8051)  # 8051 = 83 * 97
        83
    """
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    # Random starting point and constant
    x = random.randint(2, n - 1)
    c = random.randint(1, n - 1)
    y = x
    d = 1
    
    # f(x) = x² + c (mod n)
    f = lambda x: (x * x + c) % n
    
    iterations = 0
    while d == 1 and iterations < max_iterations:
        x = f(x)           # Tortoise: one step
        y = f(f(y))        # Hare: two steps
        d = gcd(abs(x - y), n)
        iterations += 1
    
    if d != n and d != 1:
        return d
    return None


def pollard_rho_brent(n: int, max_iterations: int = 1000000) -> Optional[int]:
    """
    Pollard's Rho with Brent's cycle detection improvement.
    
    Brent's algorithm is ~24% faster than Floyd's by:
    1. Moving only one pointer and comparing to saved values
    2. Accumulating GCD products to reduce expensive GCD calls
    
    The algorithm maintains a power-of-2 search pattern, saving the
    sequence value at each power of 2 and comparing subsequent values.
    
    Args:
        n: Number to factor
        max_iterations: Maximum iterations
    
    Returns:
        A non-trivial factor of n, or None
    
    Time complexity: O(n^(1/4)) expected
    
    Example:
        >>> pollard_rho_brent(1000003 * 1000033)
        1000003
    """
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    # Try a few different starting values if needed
    for _ in range(10):
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        
        g, r, q = 1, 1, 1
        ys = y
        x = y
        
        iterations = 0
        while g == 1 and iterations < max_iterations:
            x = y
            
            # Advance y by r steps
            for _ in range(r):
                y = (y * y + c) % n
            
            k = 0
            while k < r and g == 1:
                ys = y
                
                # Accumulate product for batch GCD
                # This reduces the number of expensive GCD operations
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                
                g = gcd(q, n)
                k += m
                iterations += 1
            
            r *= 2
        
        if g == n:
            # Backtrack to find exact factor
            while True:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
        
        if g != n and g != 1:
            return g
    
    return None


def pollard_rho(n: int, max_iterations: int = 1000000) -> Optional[int]:
    """
    Main Pollard Rho entry point - uses Brent's improvement.
    
    Recommended for factoring numbers in the range 10^10 to 10^25.
    For smaller numbers, trial division is faster.
    For larger numbers, ECM or MPQS may be more efficient.
    
    Args:
        n: Composite number to factor
        max_iterations: Maximum iterations per attempt
    
    Returns:
        A non-trivial factor, or None if not found
    
    Example:
        >>> n = 1000000007 * 1000000009
        >>> p = pollard_rho(n)
        >>> n % p == 0
        True
    """
    return pollard_rho_brent(n, max_iterations)


def pollard_rho_factorize(n: int) -> dict:
    """
    Completely factor n using Pollard's Rho.
    
    Recursively applies Pollard Rho until all factors are prime.
    
    Returns:
        Dict mapping prime factors to exponents
    
    Example:
        >>> pollard_rho_factorize(2 * 3 * 5 * 7 * 11 * 13)
        {2: 1, 3: 1, 5: 1, 7: 1, 11: 1, 13: 1}
    """
    if n <= 1:
        return {}
    if is_prime(n):
        return {n: 1}
    
    # Handle even numbers
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    
    if n == 1:
        return factors
    if is_prime(n):
        factors[n] = 1
        return factors
    
    # Find a factor
    factor = pollard_rho(n)
    if factor is None:
        # Fallback - shouldn't happen for composite n
        factors[n] = 1
        return factors
    
    # Recursively factor both parts
    for p, e in pollard_rho_factorize(factor).items():
        factors[p] = factors.get(p, 0) + e
    for p, e in pollard_rho_factorize(n // factor).items():
        factors[p] = factors.get(p, 0) + e
    
    return factors

"""
Fermat's Factorization Method.

Based on representing n as a difference of squares: n = a² - b² = (a+b)(a-b)

If n = pq with p ≈ q, then a ≈ √n and b is small, making this very fast.
For n = pq with p << q, this degenerates to trial division speed.

The algorithm searches for a such that a² - n is a perfect square.
Starting from a = ⌈√n⌉ and incrementing, we check if a² - n = b².

Complexity: O(|p-q| / 2) - excellent when factors are close

Best for:
- RSA moduli with poorly chosen primes (p ≈ q)
- Numbers known to have factors of similar size
- Quick check before more expensive methods

References:
    - Fermat, P. "Oeuvres" (1891) - original description
    - Knuth, D. "The Art of Computer Programming, Vol 2" - analysis
"""

from typing import Tuple, Optional
from .utils import isqrt, is_prime, gcd


def fermat_factor(n: int, max_iterations: int = 1000000) -> Tuple[int, int]:
    """
    Fermat's difference of squares factorization.
    
    Searches for a, b such that n = a² - b² = (a-b)(a+b).
    
    Args:
        n: Odd composite number to factor
        max_iterations: Maximum search iterations
    
    Returns:
        (p, q) where p * q = n, or (n, 1) if not found
    
    Example:
        >>> fermat_factor(5959)  # 59 * 101
        (59, 101)
        >>> fermat_factor(991 * 997)  # Close primes - very fast
        (991, 997)
    """
    if n <= 1:
        return (n, 1)
    if n % 2 == 0:
        return (2, n // 2)
    if is_prime(n):
        return (n, 1)
    
    # Check if n is a perfect square
    sqrt_n = isqrt(n)
    if sqrt_n * sqrt_n == n:
        # n is a perfect square - factor the root
        p, q = fermat_factor(sqrt_n, max_iterations)
        return (p, n // p)
    
    # Start with a = ceil(sqrt(n))
    a = sqrt_n + 1
    
    for _ in range(max_iterations):
        # Check if a² - n is a perfect square
        b_squared = a * a - n
        b = isqrt(b_squared)
        
        if b * b == b_squared:
            # Found! n = (a-b)(a+b)
            p = a - b
            q = a + b
            if p > 1 and q > 1 and p * q == n:
                return (min(p, q), max(p, q))
        
        a += 1
    
    return (n, 1)


def fermat_factor_optimized(n: int, max_iterations: int = 1000000) -> Tuple[int, int]:
    """
    Optimized Fermat factorization using modular arithmetic.
    
    Uses the fact that perfect squares have limited residues mod small primes
    to skip many non-square candidates quickly.
    
    Squares mod 16: {0, 1, 4, 9}
    Squares mod 9: {0, 1, 4, 7}
    Squares mod 5: {0, 1, 4}
    """
    if n <= 1:
        return (n, 1)
    if n % 2 == 0:
        return (2, n // 2)
    if is_prime(n):
        return (n, 1)
    
    sqrt_n = isqrt(n)
    if sqrt_n * sqrt_n == n:
        p, q = fermat_factor_optimized(sqrt_n, max_iterations)
        return (p, n // p)
    
    # Precompute quadratic residues for fast rejection
    qr_16 = {0, 1, 4, 9}
    qr_9 = {0, 1, 4, 7}
    qr_5 = {0, 1, 4}
    
    a = sqrt_n + 1
    
    for _ in range(max_iterations):
        b_squared = a * a - n
        
        # Quick rejection using modular arithmetic
        if (b_squared % 16) not in qr_16:
            a += 1
            continue
        if (b_squared % 9) not in qr_9:
            a += 1
            continue
        if (b_squared % 5) not in qr_5:
            a += 1
            continue
        
        # Full square check
        b = isqrt(b_squared)
        if b * b == b_squared:
            p = a - b
            q = a + b
            if p > 1 and q > 1 and p * q == n:
                return (min(p, q), max(p, q))
        
        a += 1
    
    return (n, 1)


def fermat_factor_multiplier(n: int, max_iterations: int = 100000) -> Tuple[int, int]:
    """
    Fermat with multiplier - try kn for small k to find better representation.
    
    Sometimes kn = a² - b² has smaller a - sqrt(kn), making factorization faster.
    """
    if n <= 1:
        return (n, 1)
    if n % 2 == 0:
        return (2, n // 2)
    
    # Try small multipliers
    multipliers = [1, 3, 5, 7, 11, 13, 15, 17, 19, 21, 23]
    
    for k in multipliers:
        kn = k * n
        sqrt_kn = isqrt(kn)
        
        if sqrt_kn * sqrt_kn == kn:
            continue
        
        a = sqrt_kn + 1
        
        for _ in range(max_iterations // len(multipliers)):
            b_squared = a * a - kn
            b = isqrt(b_squared)
            
            if b * b == b_squared:
                # kn = (a-b)(a+b)
                p = gcd(a - b, n)
                q = gcd(a + b, n)
                
                if 1 < p < n:
                    return (p, n // p)
                if 1 < q < n:
                    return (q, n // q)
            
            a += 1
    
    return (n, 1)

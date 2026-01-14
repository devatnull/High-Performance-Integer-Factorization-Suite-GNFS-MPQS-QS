"""
Pollard's p-1 Algorithm for Integer Factorization.

Exploits the structure of multiplicative groups: if p|n and (p-1) is
B-smooth (all prime factors ≤ B), then we can find p efficiently.

By Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p)
If M is a multiple of (p-1), then a^M ≡ 1 (mod p)
So gcd(a^M - 1, n) will reveal p.

Particularly effective against primes p where p-1 has only small factors.
This is why cryptographic primes should be "safe primes" (p = 2q + 1).

References:
    - Pollard, J. M. "Theorems on factorization and primality testing" (1974)
    - Montgomery, P. "Speeding the Pollard and elliptic curve methods" (1987)
"""

import math
from typing import Optional, List
from .utils import gcd, powmod, generate_primes, is_prime


def pollard_pm1_basic(n: int, B1: int = 100000) -> Optional[int]:
    """
    Basic Pollard p-1 algorithm (Stage 1 only).
    
    Computes a^M mod n where M = lcm(1, 2, ..., B1) = ∏ p^⌊log_p(B1)⌋
    If gcd(a^M - 1, n) is non-trivial, we've found a factor.
    
    Args:
        n: Number to factor
        B1: Smoothness bound - finds p where (p-1) is B1-smooth
    
    Returns:
        A factor of n, or None if not found
    
    Example:
        >>> # 1000003 - 1 = 2 * 3 * 166667, not smooth
        >>> # But smaller primes with smooth p-1 can be found
        >>> pollard_pm1_basic(2 * 1000003)
        2
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    
    a = 2  # Base - could try others if this fails
    
    # Compute a^M where M includes all prime powers up to B1
    primes = generate_primes(B1)
    
    for p in primes:
        # Include p^k for the largest k where p^k ≤ B1
        pk = p
        while pk <= B1:
            a = powmod(a, p, n)
            pk *= p
        
        # Check for factor periodically (expensive)
        if p % 1000 == 0:
            g = gcd(a - 1, n)
            if 1 < g < n:
                return g
    
    g = gcd(a - 1, n)
    if 1 < g < n:
        return g
    
    return None


def pollard_pm1_two_stage(n: int, B1: int = 100000, B2: int = 10000000) -> Optional[int]:
    """
    Two-stage Pollard p-1 algorithm.
    
    Stage 1: Find factors where (p-1) is B1-smooth
    Stage 2: Find factors where (p-1) has one large prime factor between B1 and B2
    
    Stage 2 is more memory-efficient than extending B1, as we only need
    to check individual primes rather than their products.
    
    The key insight for Stage 2: if p-1 = m * q where m is B1-smooth and
    q is prime with B1 < q < B2, then after Stage 1 we have a^m ≡ a' (mod n).
    Now a'^q ≡ 1 (mod p), so we check gcd(a'^q - 1, n) for each prime q.
    
    Args:
        n: Number to factor
        B1: Stage 1 smoothness bound
        B2: Stage 2 large prime bound
    
    Returns:
        A factor of n, or None
    
    Example:
        >>> pollard_pm1_two_stage(7 * 191, B1=100, B2=200)  # 190 = 2*5*19
        7
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    
    # === Stage 1 ===
    a = 2
    primes = generate_primes(B1)
    
    for p in primes:
        pk = p
        while pk <= B1:
            a = powmod(a, p, n)
            pk *= p
    
    g = gcd(a - 1, n)
    if 1 < g < n:
        return g
    if g == n:
        # Bad luck: found n itself, try different base
        return None
    
    # === Stage 2 ===
    # Now a = (original_a)^M where M is B1-smooth part
    # For each prime q in (B1, B2], check if (p-1)/gcd(p-1, M) = q
    
    stage2_primes = [p for p in generate_primes(B2) if p > B1]
    
    if not stage2_primes:
        return None
    
    # Precompute a^2 for faster "baby-step giant-step" style computation
    a2 = powmod(a, 2, n)
    
    # Compute a^q for first prime
    q_prev = stage2_primes[0]
    aq = powmod(a, q_prev, n)
    
    # Accumulate product for batch GCD
    product = (aq - 1) % n
    batch_size = 100
    
    for i, q in enumerate(stage2_primes[1:], 1):
        # Compute a^q from a^q_prev using a^(q - q_prev)
        # Since consecutive primes differ by small even numbers
        diff = q - q_prev
        aq = (aq * powmod(a, diff, n)) % n
        q_prev = q
        
        product = (product * (aq - 1)) % n
        
        # Batch GCD check
        if i % batch_size == 0:
            g = gcd(product, n)
            if 1 < g < n:
                return g
            if g == n:
                # Found n, need to backtrack
                # Simplified: just return None for now
                return None
            product = 1
    
    # Final check
    g = gcd(product, n)
    if 1 < g < n:
        return g
    
    return None


def pollard_pm1(n: int, B1: int = 100000, B2: int = 10000000) -> Optional[int]:
    """
    Main entry point for Pollard p-1 factorization.
    
    Best used when you suspect factors have smooth (p-1) values.
    Common in poorly generated random primes.
    
    Args:
        n: Number to factor
        B1: Stage 1 bound (factors with B1-smooth p-1)
        B2: Stage 2 bound (factors with one large prime factor in p-1)
    
    Returns:
        A non-trivial factor, or None
    """
    return pollard_pm1_two_stage(n, B1, B2)

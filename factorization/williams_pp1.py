"""
Williams' p+1 Factorization Algorithm.

Complementary to Pollard's p-1: finds prime factors p where (p+1) is smooth.

The algorithm uses Lucas sequences instead of simple exponentiation.
If p | n and (p+1) | M for some computed M, then the Lucas sequence
V_M will satisfy gcd(V_M - 2, n) = p (with some probability).

This is particularly effective against primes p = 2q - 1 where q is prime
(which resist p-1), but are vulnerable if p+1 = 2q has small factors.

Complexity: O(B log B log n) similar to p-1

References:
    - Williams, H.C. "A p+1 method of factoring" (1982)
    - Montgomery, P. "Speeding the Pollard and ECM methods" (1987)
"""

import random
from typing import Optional
from .utils import gcd, powmod, generate_primes, is_prime, jacobi_symbol


def lucas_sequence_v(P: int, n: int, k: int) -> int:
    """
    Compute V_k mod n using the Lucas sequence with parameter P.
    
    Lucas sequences V_k satisfy:
        V_0 = 2
        V_1 = P
        V_{k+1} = P * V_k - V_{k-1}
    
    We compute V_k efficiently using the doubling formulas:
        V_{2k} = V_k² - 2
        V_{2k+1} = V_k * V_{k+1} - P
    """
    if k == 0:
        return 2
    if k == 1:
        return P % n
    
    # Binary ladder for Lucas V sequence
    V_k = P % n
    V_k1 = (P * P - 2) % n  # V_2
    
    # Process bits of k from second-highest to lowest
    bits = bin(k)[3:]  # Skip '0b1'
    
    for bit in bits:
        if bit == '0':
            V_k1 = (V_k * V_k1 - P) % n
            V_k = (V_k * V_k - 2) % n
        else:
            V_k = (V_k * V_k1 - P) % n
            V_k1 = (V_k1 * V_k1 - 2) % n
    
    return V_k


def williams_pp1_basic(n: int, B1: int = 100000) -> Optional[int]:
    """
    Basic Williams p+1 algorithm (Stage 1).
    
    Tries random starting values P until finding a factor.
    The parameter P determines which curve we're working on.
    
    Args:
        n: Number to factor
        B1: Smoothness bound for p+1
    
    Returns:
        A factor of n if found, None otherwise
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    primes = generate_primes(B1)
    
    # Try several random starting points
    for attempt in range(20):
        # Random starting value with Jacobi symbol check
        # We want P² - 4 to be a quadratic non-residue mod p
        # (this ensures we're on a proper curve)
        P = random.randint(3, n - 2)
        
        # Skip if gcd(P² - 4, n) > 1 (we might have found factor!)
        disc = (P * P - 4) % n
        g = gcd(disc, n)
        if 1 < g < n:
            return g
        if g == n:
            continue
        
        V = P
        
        # Compute V_M where M = ∏ p^k for primes p ≤ B1
        for p in primes:
            # Compute largest power of p ≤ B1
            pk = p
            while pk * p <= B1:
                pk *= p
            
            # V = V_{pk}
            V = lucas_sequence_v(V, n, pk)
            
            # Periodic check for factor
            if p % 100 == 0:
                g = gcd(V - 2, n)
                if 1 < g < n:
                    return g
                if g == n:
                    break  # Try different P
        
        # Final check
        g = gcd(V - 2, n)
        if 1 < g < n:
            return g
    
    return None


def williams_pp1_two_stage(n: int, B1: int = 100000, 
                           B2: int = 10000000) -> Optional[int]:
    """
    Two-stage Williams p+1.
    
    Stage 1: Find factors where (p+1) is B1-smooth
    Stage 2: Find factors where (p+1) has one large prime factor in [B1, B2]
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    primes = generate_primes(B1)
    stage2_primes = [p for p in generate_primes(B2) if p > B1]
    
    for attempt in range(10):
        P = random.randint(3, n - 2)
        
        disc = (P * P - 4) % n
        g = gcd(disc, n)
        if 1 < g < n:
            return g
        if g == n:
            continue
        
        # === Stage 1 ===
        V = P
        for p in primes:
            pk = p
            while pk * p <= B1:
                pk *= p
            V = lucas_sequence_v(V, n, pk)
        
        g = gcd(V - 2, n)
        if 1 < g < n:
            return g
        if g == n:
            continue
        
        # === Stage 2 ===
        # For each prime q in (B1, B2], compute V_q and check gcd
        # Use baby-step giant-step for efficiency
        
        if not stage2_primes:
            continue
        
        # Precompute V_{2d} for small differences d
        # Consecutive primes differ by even numbers
        max_diff = 100
        V_diff = {}
        for d in range(2, max_diff + 1, 2):
            V_diff[d] = lucas_sequence_v(V, n, d)
        
        V_q = lucas_sequence_v(V, n, stage2_primes[0])
        product = (V_q - 2) % n
        
        prev_q = stage2_primes[0]
        for i, q in enumerate(stage2_primes[1:], 1):
            diff = q - prev_q
            
            if diff in V_diff:
                # V_{q} = V_{prev_q + diff} using addition formula
                # V_{m+n} = V_m * V_n - V_{m-n} ... complicated
                # Simpler: just compute directly for now
                V_q = lucas_sequence_v(V, n, q)
            else:
                V_q = lucas_sequence_v(V, n, q)
            
            prev_q = q
            product = (product * (V_q - 2)) % n
            
            # Batch GCD every 100 primes
            if i % 100 == 0:
                g = gcd(product, n)
                if 1 < g < n:
                    return g
                if g == n:
                    break
                product = 1
        
        g = gcd(product, n)
        if 1 < g < n:
            return g
    
    return None


def williams_pp1(n: int, B1: int = 100000, B2: int = 10000000) -> Optional[int]:
    """
    Williams p+1 factorization - main entry point.
    
    Finds prime factors p where (p+1) is smooth.
    Complementary to Pollard p-1 (which finds p where (p-1) is smooth).
    
    Best used in combination with p-1 for complete coverage.
    
    Args:
        n: Number to factor
        B1: Stage 1 smoothness bound
        B2: Stage 2 large prime bound
    
    Returns:
        A non-trivial factor of n, or None
    """
    return williams_pp1_two_stage(n, B1, B2)

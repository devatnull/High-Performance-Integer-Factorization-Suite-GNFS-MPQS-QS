"""
Core mathematical utilities for integer factorization.

This module provides fundamental number-theoretic functions used across
all factorization algorithms: GCD, modular arithmetic, primality testing,
and smooth number detection.
"""

import math
import random
from typing import List, Optional, Tuple

# Try to use gmpy2 for fast arbitrary precision arithmetic
try:
    import gmpy2
    from gmpy2 import mpz, is_prime as _gmpy_is_prime, gcd as _gmpy_gcd
    from gmpy2 import isqrt as _gmpy_isqrt, powmod as _gmpy_powmod
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False
    mpz = int


def gcd(a: int, b: int) -> int:
    """Euclidean algorithm for greatest common divisor."""
    if HAS_GMPY2:
        return int(_gmpy_gcd(a, b))
    while b:
        a, b = b, a % b
    return a


def isqrt(n: int) -> int:
    """Integer square root: floor(sqrt(n))."""
    if HAS_GMPY2:
        return int(_gmpy_isqrt(n))
    return math.isqrt(n)


def powmod(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod mod."""
    if HAS_GMPY2:
        return int(_gmpy_powmod(base, exp, mod))
    return pow(base, exp, mod)


def mod_inverse(a: int, m: int) -> int:
    """
    Modular multiplicative inverse using extended Euclidean algorithm.
    Returns x such that (a * x) % m == 1.
    Raises ValueError if inverse doesn't exist.
    """
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        g, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return g, x, y
    
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return (x % m + m) % m


def is_prime(n: int) -> bool:
    """
    Miller-Rabin primality test.
    
    Deterministic for n < 3,317,044,064,679,887,385,961,981 using
    specific witness bases. Probabilistic for larger numbers.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    if HAS_GMPY2:
        return _gmpy_is_prime(n) > 0
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses for deterministic test up to certain bounds
    # For n < 3,317,044,064,679,887,385,961,981
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def is_perfect_power(n: int) -> Optional[Tuple[int, int]]:
    """
    Check if n is a perfect power.
    Returns (base, exponent) if n = base^exponent, None otherwise.
    """
    if n <= 1:
        return None
    
    for k in range(2, n.bit_length() + 1):
        root = int(round(n ** (1/k)))
        # Check root and neighbors due to floating point imprecision
        for r in [root - 1, root, root + 1]:
            if r > 0 and r ** k == n:
                return (r, k)
    return None


def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).
    
    Returns:
        1 if a is a quadratic residue mod p
        -1 if a is a quadratic non-residue mod p
        0 if a ≡ 0 (mod p)
    
    Uses Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
    """
    if a % p == 0:
        return 0
    result = powmod(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """
    Tonelli-Shanks algorithm for computing square roots modulo a prime.
    
    Given n and prime p, finds r such that r² ≡ n (mod p).
    Returns None if n is not a quadratic residue mod p.
    
    Time complexity: O(log²p)
    """
    if legendre_symbol(n, p) != 1:
        return None
    
    # Factor out powers of 2 from p-1: p-1 = Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # Simple case: p ≡ 3 (mod 4)
    if S == 1:
        return powmod(n, (p + 1) // 4, p)
    
    # Find a quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    M = S
    c = powmod(z, Q, p)
    t = powmod(n, Q, p)
    R = powmod(n, (Q + 1) // 2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Update values
        b = powmod(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R


def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n) for odd n > 0.
    
    Generalization of Legendre symbol to composite moduli.
    Uses the law of quadratic reciprocity for efficient computation.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    
    a = a % n
    result = 1
    
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    
    return result if n == 1 else 0


def generate_primes(limit: int) -> List[int]:
    """
    Sieve of Eratosthenes for generating primes up to limit.
    
    Time complexity: O(n log log n)
    Space complexity: O(n)
    """
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i, is_p in enumerate(sieve) if is_p]


def factor_out_small_primes(n: int, primes: List[int]) -> Tuple[int, dict]:
    """
    Factor out small primes from n.
    
    Returns (remaining, factors) where factors is a dict mapping prime -> exponent.
    """
    factors = {}
    for p in primes:
        if p * p > n:
            break
        exp = 0
        while n % p == 0:
            n //= p
            exp += 1
        if exp > 0:
            factors[p] = exp
    return n, factors


def is_smooth(n: int, factor_base: List[int]) -> bool:
    """Check if n factors completely over the factor_base."""
    for p in factor_base:
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    return abs(n) == 1


def factor_over_base(n: int, factor_base: List[int]) -> Optional[List[int]]:
    """
    Factor n over the factor base.
    
    Returns exponent vector if n is smooth over the base, None otherwise.
    """
    exponents = []
    for p in factor_base:
        exp = 0
        while n % p == 0:
            n //= p
            exp += 1
        exponents.append(exp)
    
    if abs(n) == 1:
        return exponents
    return None


def print_progress(current: int, total: int, description: str = "") -> None:
    """Display progress bar."""
    percent = (current / total) * 100 if total > 0 else 0
    print(f"\r{description}: {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current >= total:
        print()

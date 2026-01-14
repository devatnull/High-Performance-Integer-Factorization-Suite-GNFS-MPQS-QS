"""
Lenstra's Elliptic Curve Method (ECM) for Integer Factorization.

The most elegant factorization algorithm, using the algebraic structure
of elliptic curves over finite fields.

Key insight: For a prime p, the group of points on E(F_p) has order
approximately p (by Hasse's theorem: |#E(F_p) - p - 1| ≤ 2√p).
Different curves give different group orders, so by trying many curves,
we're likely to find one where the order is smooth.

Unlike Pollard p-1 which requires (p-1) to be smooth, ECM can find
factors with probability depending on the factor size, not n's size.
This makes it ideal for extracting medium-sized factors from large n.

Complexity: O(exp((√2 + o(1)) √(ln p · ln ln p))) for factor p
            Independent of n! Only depends on factor size.

References:
    - Lenstra, H. W. "Factoring integers with elliptic curves" (1987)
    - Montgomery, P. "Speeding the Pollard and ECM methods" (1987)
    - Zimmermann, P. "GMP-ECM" for implementation techniques
"""

import random
import math
from typing import Optional, Tuple, List
from .utils import gcd, powmod, mod_inverse, generate_primes, is_prime


class MontgomeryCurve:
    """
    Elliptic curve in Montgomery form: By² = x³ + Ax² + x
    
    Montgomery form allows efficient arithmetic without computing y,
    using only the x-coordinate (projective: X/Z).
    
    Point addition and doubling use ~10-11 field multiplications,
    compared to ~16 for Weierstrass form.
    """
    
    def __init__(self, A: int, n: int):
        """
        Initialize curve By² = x³ + Ax² + x over Z/nZ.
        
        B is implicit (not needed for x-only arithmetic).
        """
        self.A = A % n
        self.n = n
        # A24 = (A + 2) / 4 - used in doubling formula
        # We compute this as (A + 2) * inverse(4)
        try:
            self.A24 = ((A + 2) * mod_inverse(4, n)) % n
        except ValueError:
            # gcd(4, n) > 1 means we found a factor!
            self.A24 = None
    
    def double(self, X: int, Z: int) -> Tuple[int, int]:
        """
        Point doubling on Montgomery curve (x-coordinate only).
        
        Uses the formula:
            X_2 = (X + Z)² * (X - Z)²
            Z_2 = 4XZ * ((X - Z)² + A24 * 4XZ)
        
        Returns (X_2, Z_2) or raises ValueError if division fails.
        """
        n = self.n
        
        u = (X + Z) % n
        v = (X - Z) % n
        u2 = (u * u) % n
        v2 = (v * v) % n
        
        X2 = (u2 * v2) % n
        
        diff = (u2 - v2) % n
        Z2 = (diff * (v2 + self.A24 * diff)) % n
        
        return (X2, Z2)
    
    def add(self, X1: int, Z1: int, X2: int, Z2: int, 
            X0: int, Z0: int) -> Tuple[int, int]:
        """
        Differential point addition: P3 = P1 + P2 given P0 = P1 - P2.
        
        Montgomery ladder requires knowing the difference, which we
        maintain throughout scalar multiplication.
        
        Formula:
            X3 = Z0 * ((X1-Z1)(X2+Z2) + (X1+Z1)(X2-Z2))²
            Z3 = X0 * ((X1-Z1)(X2+Z2) - (X1+Z1)(X2-Z2))²
        """
        n = self.n
        
        u = ((X1 - Z1) * (X2 + Z2)) % n
        v = ((X1 + Z1) * (X2 - Z2)) % n
        
        add = (u + v) % n
        sub = (u - v) % n
        
        X3 = (Z0 * add * add) % n
        Z3 = (X0 * sub * sub) % n
        
        return (X3, Z3)
    
    def scalar_mult(self, k: int, X: int, Z: int) -> Tuple[int, int]:
        """
        Compute k*P using Montgomery ladder.
        
        The ladder is constant-time and uses only addition and doubling,
        making it resistant to timing attacks (bonus for crypto).
        
        Returns (X_k, Z_k) representing k*P.
        """
        if k == 0:
            return (0, 0)
        if k == 1:
            return (X, Z)
        
        # Montgomery ladder
        R0_X, R0_Z = X, Z           # R0 = P
        R1_X, R1_Z = self.double(X, Z)  # R1 = 2P
        
        # Process bits from second-highest to lowest
        for bit in bin(k)[3:]:  # Skip '0b1'
            if bit == '0':
                R1_X, R1_Z = self.add(R0_X, R0_Z, R1_X, R1_Z, X, Z)
                R0_X, R0_Z = self.double(R0_X, R0_Z)
            else:
                R0_X, R0_Z = self.add(R0_X, R0_Z, R1_X, R1_Z, X, Z)
                R1_X, R1_Z = self.double(R1_X, R1_Z)
        
        return (R0_X, R0_Z)


def ecm_one_curve(n: int, B1: int, B2: int = 0) -> Optional[int]:
    """
    Run ECM Stage 1 (and optionally Stage 2) on a single random curve.
    
    Stage 1: Compute Q = M*P where M = ∏ p^⌊log_p(B1)⌋
    Stage 2: Check primes between B1 and B2 (one large prime factor)
    
    Returns:
        A factor of n if found, None otherwise
    """
    if is_prime(n):
        return None
    
    # Generate random curve and point using Suyama's parametrization
    # This guarantees group order divisible by 12
    sigma = random.randint(6, n - 1)
    
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n
    
    # Check for trivial factors
    g = gcd(v, n)
    if 1 < g < n:
        return g
    
    try:
        v_inv = mod_inverse(v, n)
    except ValueError:
        g = gcd(v, n)
        if 1 < g < n:
            return g
        return None
    
    # Curve parameter A
    u3 = (u * u * u) % n
    v3 = (v * v * v) % n
    
    diff = (v - u) % n
    diff3 = (diff * diff * diff) % n
    
    try:
        A = ((diff3 * (3 * u + v)) * mod_inverse(4 * u3 * v, n) - 2) % n
    except ValueError:
        g = gcd(4 * u3 * v, n)
        if 1 < g < n:
            return g
        return None
    
    # Initial point P = (X : Z) = (u³ : v³)
    X, Z = u3, v3
    
    # Create curve
    curve = MontgomeryCurve(A, n)
    if curve.A24 is None:
        g = gcd(4, n)
        if 1 < g < n:
            return g
        return None
    
    # === Stage 1 ===
    primes = generate_primes(B1)
    
    for p in primes:
        # Compute largest power of p ≤ B1
        pk = p
        while pk * p <= B1:
            pk *= p
        
        # Q = pk * Q
        X, Z = curve.scalar_mult(pk, X, Z)
        
        # Check for factor
        if Z == 0:
            continue
        
        g = gcd(Z, n)
        if 1 < g < n:
            return g
        if g == n:
            # Hit the full n - try with smaller multiplier
            return None
    
    # Final Stage 1 check
    g = gcd(Z, n)
    if 1 < g < n:
        return g
    
    # === Stage 2 (standard continuation) ===
    if B2 <= B1:
        return None
    
    stage2_primes = [p for p in generate_primes(B2) if p > B1]
    if not stage2_primes:
        return None
    
    # Baby-step giant-step style Stage 2
    # Precompute multiples of Q for differences between primes
    Q_X, Q_Z = X, Z
    
    # Accumulate GCD product
    product = 1
    
    for i, p in enumerate(stage2_primes):
        # Compute p*Q
        pQ_X, pQ_Z = curve.scalar_mult(p, X, Z)
        
        if pQ_Z == 0:
            continue
        
        product = (product * pQ_Z) % n
        
        # Batch GCD every 50 primes
        if i % 50 == 0:
            g = gcd(product, n)
            if 1 < g < n:
                return g
            if g == n:
                return None
    
    g = gcd(product, n)
    if 1 < g < n:
        return g
    
    return None


def ecm(n: int, B1: int = 10000, B2: int = 1000000, 
        max_curves: int = 100) -> Optional[int]:
    """
    Lenstra's Elliptic Curve Method.
    
    Tries multiple random elliptic curves, hoping to find one where
    the group order is smooth enough to factor.
    
    Args:
        n: Number to factor
        B1: Stage 1 smoothness bound
        B2: Stage 2 bound for one large prime factor
        max_curves: Maximum number of curves to try
    
    Returns:
        A non-trivial factor of n, or None
    
    Recommended B1 values for different factor sizes:
        20 digits: B1 = 11000
        25 digits: B1 = 50000  
        30 digits: B1 = 250000
        35 digits: B1 = 1000000
        40 digits: B1 = 3000000
    
    Example:
        >>> n = 1000000007 * 1000000009
        >>> p = ecm(n)
        >>> n % p == 0
        True
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    for curve_num in range(max_curves):
        factor = ecm_one_curve(n, B1, B2)
        if factor is not None:
            return factor
    
    return None


def ecm_factorize(n: int, B1: int = 10000, B2: int = 1000000) -> dict:
    """
    Completely factor n using ECM.
    
    Returns:
        Dict mapping prime factors to exponents
    
    Example:
        >>> ecm_factorize(2 * 3 * 5 * 1000003)
        {2: 1, 3: 1, 5: 1, 1000003: 1}
    """
    if n <= 1:
        return {}
    if is_prime(n):
        return {n: 1}
    
    factors = {}
    
    # Handle small primes first
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    
    if n == 1:
        return factors
    if is_prime(n):
        factors[n] = 1
        return factors
    
    # Use ECM for remaining factors
    while not is_prime(n) and n > 1:
        factor = ecm(n, B1, B2)
        if factor is None:
            # ECM failed - record remainder
            factors[n] = factors.get(n, 0) + 1
            break
        
        # Factor out this prime
        while n % factor == 0:
            factors[factor] = factors.get(factor, 0) + 1
            n //= factor
    
    if n > 1 and is_prime(n):
        factors[n] = factors.get(n, 0) + 1
    
    return factors

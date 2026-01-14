"""
Quadratic Sieve (QS) Algorithm for Integer Factorization.

The Quadratic Sieve finds relations x² ≡ y (mod n) where y is smooth
(factors completely over a small prime base). Combining relations where
the product of y values is a perfect square gives X² ≡ Y² (mod n),
and gcd(X ± Y, n) often reveals a factor.

Key innovations over trial division:
1. Sieve for smooth values instead of testing each individually
2. Factor base of primes where n is a quadratic residue
3. Linear algebra over GF(2) to combine relations

Complexity: L_n[1/2, 1] = exp(O(√(log n · log log n)))

References:
    - Pomerance, C. "The Quadratic Sieve Factoring Algorithm" (1984)
    - Silverman, R. "The Multiple Polynomial Quadratic Sieve" (1987)
"""

import math
from typing import List, Tuple, Optional
from .utils import (gcd, isqrt, legendre_symbol, tonelli_shanks, 
                   is_smooth, factor_over_base, generate_primes, is_prime,
                   print_progress)
from .linear_algebra import find_dependencies


def generate_factor_base(n: int, B: int) -> List[int]:
    """
    Generate the factor base: primes p ≤ B where n is a quadratic residue.
    
    We only include primes p where the Legendre symbol (n/p) = 1,
    meaning there exists x such that x² ≡ n (mod p). This ensures
    our polynomial Q(x) = x² - n can be divisible by p.
    
    Args:
        n: Number being factored
        B: Smoothness bound
    
    Returns:
        List of primes in the factor base
    """
    factor_base = [2]  # Always include 2
    
    for p in generate_primes(B + 1):
        if p == 2:
            continue
        if legendre_symbol(n % p, p) == 1:
            factor_base.append(p)
    
    return factor_base


def compute_sieve_start(n: int, factor_base: List[int], 
                        sieve_start: int) -> List[Tuple[int, int]]:
    """
    For each prime p in factor base, find starting positions for sieving.
    
    We need to solve x² ≡ n (mod p), then find which sieve positions
    are divisible by p. Returns both roots (±√n mod p).
    
    Args:
        n: Number being factored
        factor_base: List of factor base primes
        sieve_start: Starting x value for the sieve
    
    Returns:
        List of (start1, start2) pairs for each prime
    """
    starts = []
    
    for p in factor_base:
        if p == 2:
            # Handle 2 specially
            starts.append((sieve_start % 2, -1))
            continue
        
        # Find square root of n mod p
        root = tonelli_shanks(n % p, p)
        if root is None:
            starts.append((-1, -1))  # Shouldn't happen if factor base is correct
            continue
        
        # Two roots: root and p - root
        root2 = p - root
        
        # Find first sieve position divisible by p
        start1 = (root - sieve_start) % p
        start2 = (root2 - sieve_start) % p
        
        starts.append((start1, start2))
    
    return starts


def sieve_interval(n: int, factor_base: List[int], sieve_array: List[float],
                   sieve_start: int, sieve_size: int) -> None:
    """
    Logarithmic sieving: subtract log(p) for each position divisible by p.
    
    This is the key optimization of QS. Instead of trial dividing each
    Q(x), we sieve: subtract log(p) from positions divisible by p.
    Positions with small remaining log value are likely smooth.
    
    Modifies sieve_array in place.
    
    Args:
        n: Number being factored
        factor_base: Prime factor base
        sieve_array: Array to accumulate log values (will be modified)
        sieve_start: Starting x value
        sieve_size: Size of sieve interval
    """
    sqrt_n = isqrt(n)
    
    # Initialize with log|Q(x)| where Q(x) = (x + sqrt_n)² - n
    for i in range(sieve_size):
        x = sieve_start + i
        qx = (x + sqrt_n) ** 2 - n
        if qx > 0:
            sieve_array[i] = math.log(qx)
        else:
            sieve_array[i] = float('inf')  # Skip negative values
    
    # Sieve with each prime
    for p in factor_base:
        log_p = math.log(p)
        
        if p == 2:
            # Sieve by 2
            for i in range(sieve_start % 2, sieve_size, 2):
                sieve_array[i] -= log_p
            continue
        
        # Find roots of Q(x) ≡ 0 (mod p)
        root = tonelli_shanks(n % p, p)
        if root is None:
            continue
        
        # Two starting positions
        # Q(x) = (x + sqrt_n)² - n ≡ 0 (mod p)
        # x + sqrt_n ≡ ±root (mod p)
        start1 = (root - sqrt_n - sieve_start) % p
        start2 = (p - root - sqrt_n - sieve_start) % p
        
        # Sieve from both starting positions
        for start in [start1, start2]:
            for i in range(start, sieve_size, p):
                sieve_array[i] -= log_p
                
                # Also remove higher powers of p
                x = sieve_start + i
                qx = abs((x + sqrt_n) ** 2 - n)
                pk = p * p
                while qx % pk == 0:
                    sieve_array[i] -= log_p
                    pk *= p


def quadratic_sieve(n: int, factor_base: Optional[List[int]] = None,
                    sieve_size: int = 100000, 
                    time_limit: float = 300) -> List[Tuple[int, int]]:
    """
    Basic Quadratic Sieve relation collection.
    
    Collects (x, Q(x)) pairs where Q(x) = x² - n is smooth over the factor base.
    Uses logarithmic sieving for efficiency.
    
    Args:
        n: Number to factor
        factor_base: Pre-computed factor base (computed if None)
        sieve_size: Size of sieve interval
        time_limit: Maximum time in seconds
    
    Returns:
        List of (x, Q(x)) relations where Q(x) is smooth
    """
    import time
    start_time = time.time()
    
    # Compute factor base bound using L-notation heuristic
    L = math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))
    B = max(100, int(L ** 0.5))
    
    if factor_base is None:
        factor_base = generate_factor_base(n, B)
    
    sqrt_n = isqrt(n)
    relations = []
    target_relations = len(factor_base) + 20
    
    # Sieve threshold: values with log < threshold might be smooth
    threshold = sum(math.log(p) for p in factor_base[:10])  # Conservative
    
    # Sieve in intervals around sqrt(n)
    interval = 0
    while len(relations) < target_relations:
        if time.time() - start_time > time_limit:
            break
        
        # Alternate positive and negative offsets
        if interval % 2 == 0:
            sieve_start = sqrt_n + (interval // 2) * sieve_size
        else:
            sieve_start = sqrt_n - ((interval // 2) + 1) * sieve_size
            if sieve_start < 1:
                interval += 1
                continue
        
        # Initialize and sieve
        sieve_array = [0.0] * sieve_size
        sieve_interval(n, factor_base, sieve_array, sieve_start, sieve_size)
        
        # Collect smooth candidates
        for i in range(sieve_size):
            if sieve_array[i] < threshold:
                x = sieve_start + i + sqrt_n
                qx = x * x - n
                
                if qx <= 0:
                    continue
                
                # Verify smoothness with trial division
                if is_smooth(qx, factor_base):
                    relations.append((x, qx))
                    
                    if len(relations) % 10 == 0:
                        print_progress(len(relations), target_relations, "QS Relations")
                    
                    if len(relations) >= target_relations:
                        break
        
        interval += 1
    
    print()  # Newline after progress
    return relations


def qs_factor(n: int, time_limit: float = 300, verbose: bool = True) -> Tuple[int, int]:
    """
    Factor n using the Quadratic Sieve.
    
    Complete factorization pipeline:
    1. Generate factor base
    2. Sieve for smooth relations
    3. Linear algebra to find dependencies
    4. Extract factor from X² ≡ Y² (mod n)
    
    Args:
        n: Number to factor (should be composite, non-prime-power)
        time_limit: Maximum time in seconds
        verbose: Print progress information
    
    Returns:
        (factor, cofactor) tuple
    
    Example:
        >>> qs_factor(1234567)
        (127, 9721)
    """
    if n <= 1:
        return (n, 1)
    if is_prime(n):
        return (n, 1)
    
    # Compute factor base
    L = math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))
    B = max(100, int(L ** 0.5))
    factor_base = generate_factor_base(n, B)
    
    if verbose:
        print(f"Factor base size: {len(factor_base)}, B = {B}")
    
    # Collect relations
    relations = quadratic_sieve(n, factor_base, time_limit=time_limit)
    
    if verbose:
        print(f"Collected {len(relations)} relations")
    
    if len(relations) < len(factor_base) + 5:
        if verbose:
            print("Not enough relations found")
        return (n, 1)
    
    # Find dependencies using linear algebra
    dependencies = find_dependencies(relations, factor_base)
    
    if not dependencies:
        if verbose:
            print("No dependencies found")
        return (n, 1)
    
    # Try each dependency to find a factor
    for dep in dependencies:
        x_prod = 1
        total_exp = [0] * len(factor_base)
        
        for i, bit in enumerate(dep):
            if bit:
                x_prod = (x_prod * relations[i][0]) % n
                exp_vec = factor_over_base(relations[i][1], factor_base)
                if exp_vec:
                    for j, e in enumerate(exp_vec):
                        total_exp[j] += e
        
        # Compute Y from the square root of product
        y = 1
        for j, e in enumerate(total_exp):
            if e % 2 != 0:
                break  # Not a valid dependency
            y = (y * pow(factor_base[j], e // 2, n)) % n
        else:
            # Valid dependency - try to extract factor
            factor = gcd(x_prod - y, n)
            if 1 < factor < n:
                return (factor, n // factor)
            
            factor = gcd(x_prod + y, n)
            if 1 < factor < n:
                return (factor, n // factor)
    
    if verbose:
        print("Factor extraction failed")
    return (n, 1)

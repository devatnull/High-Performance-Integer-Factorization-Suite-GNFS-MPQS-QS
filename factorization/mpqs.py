"""
Multiple Polynomial Quadratic Sieve (MPQS) / Self-Initializing QS (SIQS).

MPQS improves on basic QS by using multiple polynomials. The key insight:
instead of sieving Q(x) = x² - n over a huge range, we use many polynomials
    Q_A(x) = (Ax + B)² - n  where A² divides Q_A(x)
    
This means Q_A(x)/A is an integer, and we seek smooth Q_A(x)/A values.

Self-Initializing QS (SIQS) makes polynomial switching cheap by choosing
A as a product of factor base primes, allowing fast root computation.

Complexity: L_n[1/2, 1] with better constants than basic QS

References:
    - Silverman, R. "The Multiple Polynomial Quadratic Sieve" (1987)
    - Contini, S. "Factoring integers with the self-initializing quadratic sieve"
    - Pomerance, C. "A Tale of Two Sieves" - excellent survey
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Set
from .utils import (gcd, isqrt, legendre_symbol, tonelli_shanks,
                   is_smooth, factor_over_base, generate_primes, is_prime,
                   print_progress, mod_inverse)
from .linear_algebra import find_dependencies, random_combination_search


# Optimal parameters based on digit count (from literature)
MPQS_PARAMS = {
    # digits: (factor_base_size, sieve_interval_M, num_polynomials)
    10: (100, 5000, 20),
    15: (200, 10000, 50),
    20: (400, 25000, 100),
    25: (900, 50000, 200),
    30: (1800, 65536, 400),
    35: (3600, 65536, 800),
    40: (6000, 65536, 1500),
    45: (10000, 65536, 3000),
    50: (15000, 65536, 5000),
    55: (25000, 65536, 8000),
    60: (40000, 65536, 12000),
}


def get_mpqs_params(n: int) -> Tuple[int, int, int]:
    """Get optimal MPQS parameters for n based on its size."""
    digits = len(str(n))
    
    # Find closest parameter set
    for d in sorted(MPQS_PARAMS.keys()):
        if digits <= d:
            return MPQS_PARAMS[d]
    
    # Extrapolate for very large numbers
    return (60000, 65536, 20000)


def generate_factor_base(n: int, size: int) -> Tuple[List[int], Dict[int, int]]:
    """
    Generate factor base of given size.
    
    Include primes p where n is a quadratic residue (Legendre symbol = 1).
    Also compute √n mod p for sieve initialization.
    """
    factor_base = [-1]  # Include -1 for negative values
    sqrt_n_mod_p = {-1: 0}
    
    # Always include 2
    factor_base.append(2)
    sqrt_n_mod_p[2] = n % 2
    
    p = 3
    while len(factor_base) < size + 1:
        if is_prime(p):
            if legendre_symbol(n % p, p) == 1:
                root = tonelli_shanks(n % p, p)
                if root is not None:
                    factor_base.append(p)
                    sqrt_n_mod_p[p] = root
        p += 2
    
    return factor_base, sqrt_n_mod_p


class SIQSPolynomial:
    """
    Self-Initializing QS polynomial: Q(x) = (Ax + B)² - n = A²x² + 2ABx + (B² - n)
    
    We choose A as product of primes from factor base, so A² | Q(x).
    Then Q(x)/A² is smaller and more likely to be smooth.
    """
    
    def __init__(self, A: int, B: int, n: int, 
                 factor_base: List[int], sqrt_n_mod_p: Dict[int, int]):
        self.A = A
        self.B = B
        self.n = n
        self.C = (B * B - n) // A  # Note: C = (B² - n) / A, not A²
        
        # Precompute sieve roots for each prime
        # Q(x) ≡ 0 (mod p) when Ax + B ≡ ±√n (mod p)
        self.roots: Dict[int, Tuple[int, int]] = {}
        
        for p in factor_base:
            if p == -1 or p == 2:
                continue
            if A % p == 0:
                # p divides A - single root case
                continue
            
            r = sqrt_n_mod_p.get(p)
            if r is None:
                continue
            
            try:
                A_inv = mod_inverse(A % p, p)
                root1 = ((r - B) * A_inv) % p
                root2 = ((-r - B) * A_inv) % p
                self.roots[p] = (root1, root2)
            except ValueError:
                continue
    
    def evaluate(self, x: int) -> int:
        """Evaluate Q(x) = (Ax + B)² - n."""
        Ax_B = self.A * x + self.B
        return Ax_B * Ax_B - self.n
    
    def evaluate_divided(self, x: int) -> Tuple[int, int]:
        """
        Return (Ax + B, Q(x)/A) for relation building.
        
        The relation is: (Ax + B)² ≡ A * (Q(x)/A) (mod n)
        We factor Q(x)/A over the factor base.
        """
        Ax_B = self.A * x + self.B
        Q_x = Ax_B * Ax_B - self.n
        return (Ax_B, Q_x // self.A)


def select_A_primes(factor_base: List[int], target_A: int, 
                    num_primes: int = 3) -> Optional[List[int]]:
    """
    Select primes for A = p1 * p2 * ... * pk where A ≈ target_A.
    
    Using multiple small primes gives more polynomial choices.
    """
    # Filter suitable primes (not too small, not too large)
    candidates = [p for p in factor_base if 100 < p < 50000 and p > 0]
    
    if len(candidates) < num_primes:
        candidates = [p for p in factor_base if p > 2]
    
    if len(candidates) < num_primes:
        return None
    
    # Find combination closest to target
    target_per_prime = int(target_A ** (1.0 / num_primes))
    
    # Simple greedy selection
    selected = []
    remaining_target = target_A
    
    sorted_candidates = sorted(candidates, key=lambda p: abs(p - target_per_prime))
    
    for p in sorted_candidates:
        if len(selected) >= num_primes:
            break
        if p not in selected:
            selected.append(p)
            remaining_target //= p
    
    return selected if len(selected) == num_primes else None


def compute_B_values(A_primes: List[int], n: int, 
                     sqrt_n_mod_p: Dict[int, int]) -> List[int]:
    """
    Compute valid B values for A = ∏ A_primes using CRT.
    
    We need B² ≡ n (mod A), which has 2^k solutions for k primes.
    """
    # Compute B mod each prime using Tonelli-Shanks roots
    B_mod_primes = []
    for p in A_primes:
        r = sqrt_n_mod_p.get(p)
        if r is None:
            return []
        B_mod_primes.append((r, p))
    
    # Generate all 2^k combinations of ±roots
    k = len(A_primes)
    B_values = []
    
    for signs in range(1 << k):
        # Build B using CRT
        residues = []
        moduli = []
        for i, (r, p) in enumerate(B_mod_primes):
            if signs & (1 << i):
                residues.append(p - r)  # Negative root
            else:
                residues.append(r)
            moduli.append(p)
        
        # CRT reconstruction
        B = chinese_remainder_theorem(residues, moduli)
        if B is not None:
            B_values.append(B)
    
    return B_values


def chinese_remainder_theorem(residues: List[int], moduli: List[int]) -> Optional[int]:
    """Compute x such that x ≡ residues[i] (mod moduli[i]) for all i."""
    if not residues:
        return None
    
    result = residues[0]
    mod = moduli[0]
    
    for i in range(1, len(residues)):
        r, m = residues[i], moduli[i]
        
        # Find x ≡ result (mod mod) and x ≡ r (mod m)
        try:
            mod_inv = mod_inverse(mod % m, m)
        except ValueError:
            return None
        
        result = result + mod * ((r - result) * mod_inv % m)
        mod = mod * m
    
    return result % mod


def sieve_polynomial(poly: SIQSPolynomial, factor_base: List[int],
                     M: int) -> List[Tuple[int, int]]:
    """
    Sieve one polynomial over interval [-M, M].
    
    Uses logarithmic sieving: initialize with log|Q(x)|, subtract log(p)
    at positions divisible by p. Small residuals indicate smooth values.
    """
    sieve_size = 2 * M
    sieve_log = [0.0] * sieve_size
    
    # Initialize with log|Q(x)/A|
    A = poly.A
    for i in range(sieve_size):
        x = i - M
        _, Q_div_A = poly.evaluate_divided(x)
        if Q_div_A > 0:
            sieve_log[i] = math.log(Q_div_A)
        elif Q_div_A < 0:
            sieve_log[i] = math.log(-Q_div_A)
        else:
            sieve_log[i] = 0
    
    # Sieve with each prime
    for p in factor_base:
        if p <= 0:
            continue
        if p not in poly.roots:
            continue
        
        log_p = math.log(p)
        root1, root2 = poly.roots[p]
        
        # Sieve from both roots
        start1 = (root1 + M) % p
        start2 = (root2 + M) % p
        
        for i in range(start1, sieve_size, p):
            sieve_log[i] -= log_p
        
        if start2 != start1:
            for i in range(start2, sieve_size, p):
                sieve_log[i] -= log_p
    
    # Collect smooth candidates
    # Threshold: values with small log residual are likely smooth
    threshold = math.log(factor_base[-1]) * 2.5 if factor_base[-1] > 0 else 10
    
    relations = []
    for i in range(sieve_size):
        if sieve_log[i] < threshold:
            x = i - M
            Ax_B, Q_div_A = poly.evaluate_divided(x)
            
            if Q_div_A == 0:
                continue
            
            # Verify smoothness with trial division
            if is_smooth(abs(Q_div_A), [p for p in factor_base if p > 0]):
                # Include A in the factorization since relation is:
                # (Ax+B)² ≡ A * (Q/A) (mod n)
                relations.append((Ax_B, abs(Q_div_A) * A))
    
    return relations


def siqs(n: int, time_limit: float = 300, verbose: bool = True) -> List[Tuple[int, int]]:
    """
    Self-Initializing Quadratic Sieve - main relation collection.
    """
    import time
    start_time = time.time()
    
    # Get optimal parameters
    fb_size, M, max_polys = get_mpqs_params(n)
    
    # Generate factor base
    factor_base, sqrt_n_mod_p = generate_factor_base(n, fb_size)
    target_relations = len(factor_base) + 20
    
    if verbose:
        digits = len(str(n))
        print(f"SIQS: {digits} digits, FB={len(factor_base)}, M={M}, target={target_relations}")
    
    relations: List[Tuple[int, int]] = []
    used_A: Set[int] = set()
    
    # Target A size: √(2n) / M for optimal polynomial values
    sqrt_2n = isqrt(2 * n)
    target_A = sqrt_2n // M
    
    # Number of primes in A (more primes = more B choices but larger A)
    num_A_primes = 3 if len(str(n)) < 30 else 4
    
    poly_count = 0
    while len(relations) < target_relations and poly_count < max_polys:
        if time.time() - start_time > time_limit * 0.9:
            break
        
        # Select primes for A
        A_primes = select_A_primes(factor_base, target_A, num_A_primes)
        if A_primes is None:
            # Fallback: random selection
            candidates = [p for p in factor_base if p > 10]
            if len(candidates) >= num_A_primes:
                A_primes = random.sample(candidates, num_A_primes)
            else:
                break
        
        A = 1
        for p in A_primes:
            A *= p
        
        if A in used_A:
            continue
        used_A.add(A)
        
        # Compute all valid B values
        B_values = compute_B_values(A_primes, n, sqrt_n_mod_p)
        
        for B in B_values:
            if time.time() - start_time > time_limit * 0.9:
                break
            if len(relations) >= target_relations:
                break
            
            # Create and sieve polynomial
            poly = SIQSPolynomial(A, B, n, factor_base, sqrt_n_mod_p)
            new_rels = sieve_polynomial(poly, factor_base, M)
            relations.extend(new_rels)
            poly_count += 1
            
            if verbose and poly_count % 50 == 0:
                print_progress(len(relations), target_relations, 
                             f"SIQS ({poly_count} polys)")
    
    if verbose:
        print()
    
    return relations


def mpqs_factor(n: int, time_limit: float = 300, verbose: bool = True) -> Tuple[int, int]:
    """
    Factor n using MPQS/SIQS.
    
    Args:
        n: Number to factor (works best for 20-60 digit semiprimes)
        time_limit: Maximum time in seconds
        verbose: Print progress
    
    Returns:
        (p, q) where p * q = n, or (n, 1) if failed
    """
    import time
    start_time = time.time()
    
    if n <= 1:
        return (n, 1)
    if is_prime(n):
        return (n, 1)
    
    # Quick trial division
    for p in generate_primes(100000):
        if n % p == 0:
            return (p, n // p)
    
    # Collect relations using SIQS
    relations = siqs(n, time_limit * 0.8, verbose)
    
    # Get factor base for linear algebra
    fb_size, _, _ = get_mpqs_params(n)
    factor_base, _ = generate_factor_base(n, fb_size)
    factor_base = [p for p in factor_base if p > 0]  # Remove -1
    
    if verbose:
        print(f"Collected {len(relations)} relations")
    
    if len(relations) < len(factor_base) // 2:
        if verbose:
            print("Insufficient relations")
        return (n, 1)
    
    # Try random combinations first
    if verbose:
        print("Searching for factor...")
    
    factor = random_combination_search(relations, factor_base, n, max_attempts=2000)
    if factor:
        return (factor, n // factor)
    
    # Full linear algebra
    if verbose:
        print("Running linear algebra...")
    
    dependencies = find_dependencies(relations, factor_base)
    
    if not dependencies:
        if verbose:
            print("No dependencies found")
        return (n, 1)
    
    # Extract factor
    for dep in dependencies[:50]:
        x_prod = 1
        total_exp = [0] * len(factor_base)
        
        for i, bit in enumerate(dep):
            if bit and i < len(relations):
                x_val, y_sq = relations[i]
                x_prod = (x_prod * x_val) % n
                
                exp_vec = factor_over_base(y_sq, factor_base)
                if exp_vec:
                    for j, e in enumerate(exp_vec):
                        total_exp[j] += e
        
        if not all(e % 2 == 0 for e in total_exp):
            continue
        
        y = 1
        for j, e in enumerate(total_exp):
            y = (y * pow(factor_base[j], e // 2, n)) % n
        
        for candidate in [x_prod - y, x_prod + y]:
            factor = gcd(candidate, n)
            if 1 < factor < n:
                return (factor, n // factor)
    
    if verbose:
        print("Factor extraction failed")
    return (n, 1)

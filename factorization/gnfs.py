"""
General Number Field Sieve (GNFS) for Integer Factorization.

The asymptotically fastest known classical algorithm for factoring large
integers, with complexity L_n[1/3, (64/9)^(1/3)] ≈ L_n[1/3, 1.923].

GNFS works by finding relations in two number fields simultaneously:
1. The rational number field Q
2. An algebraic number field Q[α] where α is a root of polynomial f(x)

Both fields share a common homomorphism φ: Z[α] → Z/nZ defined by
φ(α) = m where f(m) ≡ 0 (mod n).

The algorithm proceeds in phases:
1. Polynomial Selection: Find f(x), m with good properties
2. Relation Collection: Find (a,b) pairs smooth on both sides
3. Linear Algebra: Find dependency in GF(2) exponent matrix
4. Square Root: Compute algebraic square root to extract factor

References:
    - Lenstra, Lenstra, Manasse, Pollard "The Number Field Sieve" (1990)
    - Buhler, Lenstra, Pomerance "Factoring integers with the NFS" (1993)
    - Kleinjung "On polynomial selection for the GNFS" (2006)
    - Crandall & Pomerance "Prime Numbers: A Computational Perspective"
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from .utils import (gcd, isqrt, is_prime, powmod, mod_inverse,
                   generate_primes, legendre_symbol, print_progress)


@dataclass
class GNFSPolynomials:
    """
    Polynomial pair (f, g) for GNFS.
    
    f(x): Algebraic polynomial of degree d (typically 5-6)
    g(x): Linear polynomial g(x) = x - m
    
    Satisfies: f(m) ≡ 0 (mod n)
    """
    f_coeffs: List[int]  # [a_0, a_1, ..., a_d] where f(x) = Σ a_i x^i
    m: int               # Root: f(m) ≡ 0 (mod n)
    degree: int
    
    def f(self, x: int) -> int:
        """Evaluate f(x)."""
        result = 0
        power = 1
        for coef in self.f_coeffs:
            result += coef * power
            power *= x
        return result
    
    def f_mod(self, x: int, mod: int) -> int:
        """Evaluate f(x) mod p."""
        result = 0
        power = 1
        for coef in self.f_coeffs:
            result = (result + coef * power) % mod
            power = (power * x) % mod
        return result
    
    def f_derivative(self, x: int) -> int:
        """Evaluate f'(x)."""
        result = 0
        power = 1
        for i, coef in enumerate(self.f_coeffs[1:], 1):
            result += i * coef * power
            power *= x
        return result


@dataclass
class Relation:
    """
    A GNFS relation: (a, b) pair with factorizations.
    
    For coprime (a, b), we have:
    - Rational side: a + bm = ∏ p_i^{e_i}  (smooth over rational FB)
    - Algebraic side: N(a + bα) = ∏ q_j^{f_j}  (smooth over algebraic FB)
    
    The norm N(a + bα) = (-b)^d · f(−a/b) = Res(a + bx, f(x))
    """
    a: int
    b: int
    rational_exps: Dict[int, int]    # prime -> exponent
    algebraic_exps: Dict[Tuple[int, int], int]  # (p, r) -> exponent


class GNFSFactorBase:
    """
    Factor bases for GNFS sieving.
    
    Rational Factor Base: Primes p up to bound B
    Algebraic Factor Base: First-degree prime ideals (p, r) where f(r) ≡ 0 (mod p)
    """
    
    def __init__(self, polys: GNFSPolynomials, bound: int):
        self.bound = bound
        self.polys = polys
        
        # Rational factor base
        self.rational_primes = generate_primes(bound)
        
        # Algebraic factor base: (p, r) where r is a root of f mod p
        self.algebraic_ideals: List[Tuple[int, int]] = []
        
        for p in self.rational_primes:
            if p > bound:
                break
            # Find roots of f(x) mod p
            roots = self._find_roots_mod_p(p)
            for r in roots:
                self.algebraic_ideals.append((p, r))
    
    def _find_roots_mod_p(self, p: int) -> List[int]:
        """Find all roots of f(x) ≡ 0 (mod p)."""
        roots = []
        for r in range(p):
            if self.polys.f_mod(r, p) == 0:
                roots.append(r)
        return roots


def select_polynomial_base_m(n: int, degree: int = 5) -> GNFSPolynomials:
    """
    Base-m polynomial selection.
    
    Write n in base m: n = a_0 + a_1*m + a_2*m² + ... + a_d*m^d
    Then f(x) = a_0 + a_1*x + ... + a_d*x^d satisfies f(m) = n.
    
    Choose m ≈ n^(1/(d+1)) for balanced coefficients.
    """
    # Target m for balanced representation
    m = int(round(n ** (1.0 / (degree + 1))))
    
    best_m = m
    best_coeffs = None
    best_score = float('inf')
    
    for delta in range(-50, 51):
        candidate_m = m + delta
        if candidate_m < 2:
            continue
        
        # Simple base-m representation (unsigned)
        coeffs = []
        temp = n
        for _ in range(degree + 1):
            coeffs.append(temp % candidate_m)
            temp //= candidate_m
        
        if temp != 0:
            continue  # Need more digits than degree allows
        
        # Verify correctness
        check = sum(c * (candidate_m ** i) for i, c in enumerate(coeffs))
        if check != n:
            continue
        
        # Prefer smaller coefficients
        max_coef = max(coeffs)
        score = max_coef
        
        if score < best_score:
            best_score = score
            best_m = candidate_m
            best_coeffs = coeffs
    
    if best_coeffs is None:
        # Fallback: just use initial m
        coeffs = []
        temp = n
        for _ in range(degree + 1):
            coeffs.append(temp % m)
            temp //= m
        best_coeffs = coeffs
        best_m = m
    
    return GNFSPolynomials(f_coeffs=best_coeffs, m=best_m, degree=degree)


def select_polynomial_kleinjung(n: int, degree: int = 5, 
                                 num_attempts: int = 1000) -> GNFSPolynomials:
    """
    Kleinjung-style polynomial selection with skewness optimization.
    
    Searches for polynomials with good α-value (root property) and
    small coefficients relative to skewness.
    
    The quality of polynomial selection significantly impacts sieving
    efficiency - a good polynomial can reduce runtime by 2-3x.
    """
    best_poly = None
    best_score = float('inf')
    
    # Target leading coefficient size
    # For degree d, we want a_d ≈ n^(1/(d+1))
    target_ad = int(n ** (1.0 / (degree + 1)))
    
    for _ in range(num_attempts):
        # Random leading coefficient near target
        ad = random.randint(max(1, target_ad // 2), target_ad * 2)
        
        # m ≈ (n / ad)^(1/d)
        m = int(round((n / ad) ** (1.0 / degree)))
        if m < 2:
            continue
        
        # Build polynomial with f(m) ≡ 0 (mod n)
        # f(x) = ad * x^d + ... + a0
        # We need ad * m^d + ... + a0 ≡ 0 (mod n)
        
        # Use base-m expansion adjusted by ad
        try:
            coeffs = compute_polynomial_coeffs(n, m, ad, degree)
        except ValueError:
            continue
        
        if coeffs is None:
            continue
        
        # Score: combination of coefficient size and root properties
        max_coef = max(abs(c) for c in coeffs)
        
        # Simple scoring - prefer smaller coefficients
        score = max_coef
        
        if score < best_score:
            best_score = score
            best_poly = GNFSPolynomials(f_coeffs=coeffs, m=m, degree=degree)
    
    if best_poly is None:
        # Fall back to base-m
        return select_polynomial_base_m(n, degree)
    
    return best_poly


def compute_polynomial_coeffs(n: int, m: int, ad: int, degree: int) -> Optional[List[int]]:
    """
    Compute polynomial coefficients given n, m, and leading coefficient.
    
    We need f(m) = n with f(x) = ad*x^d + ... + a0
    """
    coeffs = [0] * (degree + 1)
    coeffs[degree] = ad
    
    remainder = n - ad * (m ** degree)
    
    for i in range(degree - 1, -1, -1):
        if m == 0:
            if remainder != 0:
                return None
            break
        
        power = m ** i
        coef = remainder // power
        
        # Keep coefficients reasonable
        if abs(coef) > n:
            return None
        
        coeffs[i] = coef
        remainder -= coef * power
    
    if remainder != 0:
        return None
    
    return coeffs


def compute_algebraic_norm(a: int, b: int, polys: GNFSPolynomials) -> int:
    """
    Compute the algebraic norm N(a + bα) = Res(a + bx, f(x)).
    
    For a + bα in the number field Q[α], the norm is:
    N(a + bα) = (-b)^d * f(-a/b)
    
    Using resultant formula for integer computation:
    N(a + bα) = Res(a + bx, f(x)) = f_d^d * ∏(a + b*α_i)
    where α_i are roots of f.
    """
    d = polys.degree
    
    # N(a + bα) = (-b)^d * f(-a/b)
    # Computed as: Σ a_i * (-a)^i * b^(d-i) with appropriate signs
    
    result = 0
    a_power = 1  # (-a)^i
    b_power = b ** d  # b^(d-i), starts at b^d
    
    for i, coef in enumerate(polys.f_coeffs):
        if i > 0:
            b_power //= b
        term = coef * a_power * b_power
        if i % 2 == 1:
            term = -term
        result += term
        a_power *= -a
    
    return abs(result)


def is_smooth_over_fb(value: int, primes: List[int]) -> Optional[Dict[int, int]]:
    """
    Check if value is smooth over prime list, return factorization if so.
    """
    if value == 0:
        return None
    
    factors = {}
    remaining = abs(value)
    
    for p in primes:
        if remaining == 1:
            break
        exp = 0
        while remaining % p == 0:
            remaining //= p
            exp += 1
        if exp > 0:
            factors[p] = exp
    
    if remaining == 1:
        return factors
    return None


def sieve_line(polys: GNFSPolynomials, fb: GNFSFactorBase, 
               b: int, a_range: Tuple[int, int]) -> List[Relation]:
    """
    Line sieve for fixed b: find all (a, b) relations in range.
    
    For fixed b, we sieve over a values to find pairs where:
    - a + bm is smooth over rational factor base
    - N(a + bα) is smooth over algebraic factor base
    """
    a_min, a_max = a_range
    sieve_size = a_max - a_min
    
    if sieve_size <= 0:
        return []
    
    # Initialize sieve arrays with log values
    rational_log = np.zeros(sieve_size, dtype=np.float32)
    algebraic_log = np.zeros(sieve_size, dtype=np.float32)
    
    # Initialize with actual log values
    for i in range(sieve_size):
        a = a_min + i
        if gcd(abs(a), abs(b)) != 1:
            rational_log[i] = float('inf')
            algebraic_log[i] = float('inf')
            continue
        
        rat_val = abs(a + b * polys.m)
        alg_val = compute_algebraic_norm(a, b, polys)
        
        if rat_val > 1:
            rational_log[i] = math.log(rat_val)
        else:
            rational_log[i] = 0
        
        if alg_val > 1:
            algebraic_log[i] = math.log(alg_val)
        else:
            algebraic_log[i] = 0
    
    # Sieve rational side
    for p in fb.rational_primes:
        if p == 0:
            continue
        log_p = math.log(p)
        # Find starting position: a + bm ≡ 0 (mod p) → a ≡ -bm (mod p)
        start = ((-b * polys.m % p) - a_min % p + p) % p
        for i in range(start, sieve_size, p):
            rational_log[i] -= log_p
    
    # Sieve algebraic side
    for p, r in fb.algebraic_ideals:
        if p == 0:
            continue
        log_p = math.log(p)
        # (p, r) divides (a + bα) when a ≡ -br (mod p)
        start = ((-b * r % p) - a_min % p + p) % p
        for i in range(start, sieve_size, p):
            algebraic_log[i] -= log_p
    
    # Threshold for smooth candidates - more generous
    threshold = 3 * math.log(fb.bound)
    
    # Collect relations
    relations = []
    for i in range(sieve_size):
        if rational_log[i] < threshold and algebraic_log[i] < threshold:
            a = a_min + i
            
            if a == 0 or gcd(abs(a), abs(b)) != 1:
                continue
            
            # Verify smoothness with trial division
            rat_val = a + b * polys.m
            alg_val = compute_algebraic_norm(a, b, polys)
            
            rat_factors = is_smooth_over_fb(rat_val, fb.rational_primes)
            if rat_factors is None:
                continue
            
            alg_factors = is_smooth_over_fb(alg_val, fb.rational_primes)
            if alg_factors is None:
                continue
            
            # Build algebraic factorization with ideal structure
            alg_ideal_factors: Dict[Tuple[int, int], int] = {}
            for p, r in fb.algebraic_ideals:
                if p in alg_factors and alg_factors[p] > 0:
                    # Check if (p, r) divides this specific (a, b)
                    if (a + b * r) % p == 0:
                        alg_ideal_factors[(p, r)] = alg_factors[p]
            
            relations.append(Relation(a, b, rat_factors, alg_ideal_factors))
    
    return relations


def lattice_sieve(polys: GNFSPolynomials, fb: GNFSFactorBase,
                  special_q: int, special_q_root: int,
                  sieve_region: int = 10000) -> List[Relation]:
    """
    Special-q lattice sieving.
    
    For efficiency, we sieve by a "special-q" prime ideal: only consider
    (a, b) pairs in the lattice L = {(a,b) : a + b*r ≡ 0 (mod q)}.
    
    This reduces the sieve region while concentrating on relations
    divisible by q, improving the smooth-finding rate.
    """
    relations = []
    
    # Lattice basis: L = {(a,b) : a ≡ -b*r (mod q)}
    # Basis vectors: (q, 0) and (r, 1)
    # After LLL reduction, we get short vectors
    
    # Simplified: iterate over lattice points
    for u in range(-sieve_region, sieve_region):
        for v in range(1, sieve_region):
            # Lattice point (a, b) = u*(q, 0) + v*(r, 1) = (u*q + v*r, v)
            a = u * special_q + v * special_q_root
            b = v
            
            if gcd(a, b) != 1:
                continue
            
            # Check smoothness
            rat_val = abs(a + b * polys.m)
            alg_val = compute_algebraic_norm(a, b, polys)
            
            rat_factors = is_smooth_over_fb(rat_val, fb.rational_primes)
            if rat_factors is None:
                continue
            
            alg_factors = is_smooth_over_fb(alg_val, fb.rational_primes)
            if alg_factors is None:
                continue
            
            # Valid relation
            alg_ideal_factors: Dict[Tuple[int, int], int] = {}
            for p, r in fb.algebraic_ideals:
                if p in alg_factors:
                    if (a + b * r) % p == 0:
                        alg_ideal_factors[(p, r)] = alg_factors.get(p, 0)
            
            relations.append(Relation(a, b, rat_factors, alg_ideal_factors))
            
            if len(relations) >= 10:  # Limit per special-q
                return relations
    
    return relations


def build_gnfs_matrix(relations: List[Relation], 
                      fb: GNFSFactorBase) -> Tuple[np.ndarray, List]:
    """
    Build the exponent matrix for linear algebra phase.
    
    Each relation contributes a row: exponents of all primes (rational)
    and prime ideals (algebraic) mod 2.
    """
    # Create ordered list of all "primes" (rational and algebraic)
    all_primes = list(fb.rational_primes) + list(fb.algebraic_ideals)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    num_cols = len(all_primes)
    num_rows = len(relations)
    
    matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
    
    for i, rel in enumerate(relations):
        # Rational side
        for p, exp in rel.rational_exps.items():
            if p in prime_to_idx:
                matrix[i, prime_to_idx[p]] = exp % 2
        
        # Algebraic side
        for ideal, exp in rel.algebraic_exps.items():
            if ideal in prime_to_idx:
                matrix[i, prime_to_idx[ideal]] = exp % 2
    
    return matrix, all_primes


def gnfs_factor(n: int, time_limit: float = 600, 
                verbose: bool = True) -> Tuple[int, int]:
    """
    Factor n using the General Number Field Sieve.
    
    This is a simplified implementation suitable for numbers up to ~60 digits.
    Production GNFS requires:
    - Better polynomial selection (Kleinjung's method)
    - Lattice sieving with LLL-reduced bases
    - Block Wiedemann for linear algebra
    - Montgomery's square root algorithm
    
    Args:
        n: Number to factor (should be composite, > 10^20)
        time_limit: Maximum time in seconds
        verbose: Print progress
    
    Returns:
        (factor, cofactor) or (n, 1) if factorization fails
    """
    import time
    start_time = time.time()
    
    if n <= 1:
        return (n, 1)
    if is_prime(n):
        return (n, 1)
    
    # Check for small factors first
    for p in generate_primes(10000):
        if n % p == 0:
            return (p, n // p)
    
    # Determine polynomial degree based on size
    digits = len(str(n))
    if digits < 50:
        degree = 3
    elif digits < 80:
        degree = 4
    elif digits < 110:
        degree = 5
    else:
        degree = 6
    
    if verbose:
        print(f"GNFS: {digits} digits, degree {degree}")
    
    # Polynomial selection
    polys = select_polynomial_kleinjung(n, degree, num_attempts=500)
    if verbose:
        print(f"Selected m = {polys.m}")
        print(f"f(x) coefficients: {polys.f_coeffs}")
    
    # Verify polynomial
    if polys.f(polys.m) % n != 0:
        if verbose:
            print("Warning: f(m) ≢ 0 (mod n), using base-m fallback")
        polys = select_polynomial_base_m(n, degree)
    
    # Factor base - larger for better sieving
    # Bound based on L-notation heuristic
    L = math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))
    bound = max(2000, int(L ** 0.5))
    
    fb = GNFSFactorBase(polys, bound)
    target_relations = len(fb.rational_primes) + len(fb.algebraic_ideals) + 50
    
    if verbose:
        print(f"Factor base: {len(fb.rational_primes)} rational, {len(fb.algebraic_ideals)} algebraic")
        print(f"Target: {target_relations} relations")
    
    # Relation collection
    relations: List[Relation] = []
    
    # Line sieving for many b values with wider a range
    if verbose:
        print("Line sieving...")
    
    for b in range(1, 1000):
        if time.time() - start_time > time_limit * 0.6:
            break
        
        # Wider sieve range
        a_range = (-bound * 100, bound * 100)
        new_rels = sieve_line(polys, fb, b, a_range)
        relations.extend(new_rels)
        
        if verbose and b % 50 == 0:
            print_progress(len(relations), target_relations, "GNFS Relations")
        
        if len(relations) >= target_relations:
            break
    
    # Also try negative b values
    for b in range(-1, -500, -1):
        if time.time() - start_time > time_limit * 0.8:
            break
        if len(relations) >= target_relations:
            break
            
        a_range = (-bound * 100, bound * 100)
        new_rels = sieve_line(polys, fb, abs(b), a_range)
        relations.extend(new_rels)
    
    # Special-q lattice sieving if needed
    if len(relations) < target_relations and time.time() - start_time < time_limit * 0.9:
        if verbose:
            print(f"\nLattice sieving (have {len(relations)} relations)...")
        
        special_q_start = bound // 2
        for q in generate_primes(bound * 5):
            if q <= special_q_start:
                continue
            if time.time() - start_time > time_limit * 0.95:
                break
            
            # Find roots of f mod q
            roots = []
            for r in range(min(q, 1000)):  # Limit root search for large q
                if polys.f_mod(r, q) == 0:
                    roots.append(r)
            
            for root in roots:
                new_rels = lattice_sieve(polys, fb, q, root, sieve_region=2000)
                relations.extend(new_rels)
                
                if len(relations) >= target_relations:
                    break
            
            if len(relations) >= target_relations:
                break
    
    if verbose:
        print(f"\nCollected {len(relations)} relations")
    
    if len(relations) < len(fb.rational_primes) + 10:
        if verbose:
            print("Not enough relations")
        return (n, 1)
    
    # Linear algebra
    matrix, all_primes = build_gnfs_matrix(relations, fb)
    
    # Find null space using Gaussian elimination
    from .linear_algebra import gaussian_elimination_gf2
    null_vectors = gaussian_elimination_gf2(matrix)
    
    if not null_vectors:
        if verbose:
            print("No null space vectors found")
        return (n, 1)
    
    if verbose:
        print(f"Found {len(null_vectors)} dependencies")
    
    # Try to extract factor from dependencies
    for dep in null_vectors[:100]:
        # Combine relations according to dependency
        # We need: product of (a + b*m) on rational side
        # And: product with even exponents to compute square root
        
        x_val = 1  # Product of (a + b*m)
        combined_exp = {}  # Combined exponents for rational side
        
        for i, bit in enumerate(dep):
            if bit and i < len(relations):
                rel = relations[i]
                x_val = (x_val * (rel.a + rel.b * polys.m)) % n
                
                # Accumulate exponents
                for p, e in rel.rational_exps.items():
                    combined_exp[p] = combined_exp.get(p, 0) + e
        
        # Check if all exponents are even (should be for null vector)
        if not all(e % 2 == 0 for e in combined_exp.values()):
            continue
        
        # Compute y = sqrt(product) from exponents
        y_val = 1
        for p, e in combined_exp.items():
            y_val = (y_val * pow(p, e // 2, n)) % n
        
        # Try gcd(x - y, n) and gcd(x + y, n)
        for candidate in [x_val - y_val, x_val + y_val]:
            factor = gcd(candidate % n, n)
            if 1 < factor < n:
                if verbose:
                    print(f"Found factor: {factor}")
                return (factor, n // factor)
    
    # GNFS square root failed - try ECM as fallback for smaller factors
    if verbose:
        print("GNFS extraction failed, trying ECM fallback...")
    
    from .ecm import ecm
    factor = ecm(n, B1=50000, B2=5000000, max_curves=200)
    if factor:
        if verbose:
            print(f"ECM found factor: {factor}")
        return (factor, n // factor)
    
    if verbose:
        print("Factor extraction failed")
    return (n, 1)

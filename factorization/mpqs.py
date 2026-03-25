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
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from .utils import (gcd, isqrt, legendre_symbol, tonelli_shanks,
                   generate_primes, is_prime,
                   print_progress, mod_inverse)
from .linear_algebra import find_dependencies, prefactor_relations, random_combination_search
from .simd import factor_over_base_numba, numba_available, sieve_with_numba


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


@dataclass(frozen=True)
class MPQSFactorBaseContext:
    """Shared factor-base metadata reused across the full MPQS run."""
    factor_base: List[int]
    positive_base: List[int]
    sqrt_n_mod_p: Dict[int, int]
    prime_to_index: Dict[int, int]
    log_by_prime: Dict[int, float]
    threshold: float
    use_numba: bool


def get_mpqs_params(n: int) -> Tuple[int, int, int]:
    """Get optimal MPQS parameters for n based on its size."""
    digits = len(str(n))
    
    # Find closest parameter set
    for d in sorted(MPQS_PARAMS.keys()):
        if digits <= d:
            return MPQS_PARAMS[d]
    
    # Extrapolate for very large numbers
    return (60000, 65536, 20000)


def _nth_prime_upper_bound(n: int) -> int:
    """Upper bound for the nth prime used to size sieve-backed generation."""
    if n < 6:
        return 15
    return int(n * (math.log(n) + math.log(math.log(n)))) + 10


def generate_factor_base(n: int, size: int) -> Tuple[List[int], Dict[int, int]]:
    """
    Generate factor base of given size.
    
    Include primes p where n is a quadratic residue (Legendre symbol = 1).
    Also compute √n mod p for sieve initialization.
    """
    factor_base = [-1, 2]
    sqrt_n_mod_p = {-1: 0, 2: n % 2}
    target_size = size + 1
    prime_upper = _nth_prime_upper_bound(max(32, size * 6))
    processed = 0

    while len(factor_base) < target_size:
        primes = generate_primes(prime_upper)
        new_primes = primes[processed:]
        processed = len(primes)

        for p in new_primes:
            if p == 2:
                continue
            residue = n % p
            if legendre_symbol(residue, p) != 1:
                continue

            root = tonelli_shanks(residue, p)
            if root is None:
                continue

            factor_base.append(p)
            sqrt_n_mod_p[p] = root
            if len(factor_base) >= target_size:
                break

        prime_upper *= 2

    return factor_base, sqrt_n_mod_p


def _build_factor_base_context(n: int, size: int) -> MPQSFactorBaseContext:
    """Construct reusable factor-base metadata for MPQS sieving and extraction."""
    factor_base, sqrt_n_mod_p = generate_factor_base(n, size)
    positive_base = [p for p in factor_base if p > 0]
    return MPQSFactorBaseContext(
        factor_base=factor_base,
        positive_base=positive_base,
        sqrt_n_mod_p=sqrt_n_mod_p,
        prime_to_index={p: i for i, p in enumerate(positive_base)},
        log_by_prime={p: math.log(p) for p in positive_base},
        threshold=math.log(positive_base[-1]) * 2.5 if positive_base else 10.0,
        use_numba=numba_available(),
    )


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


def _factor_relation_value(
    q_div_a: int,
    a_primes: List[int],
    context: MPQSFactorBaseContext,
) -> Optional[List[int]]:
    """Factor ``|Q(x)/A| * A`` over the positive factor base."""
    exp_vec = factor_over_base_numba(abs(q_div_a), context.positive_base)
    if exp_vec is None:
        return None

    for p in a_primes:
        index = context.prime_to_index.get(p)
        if index is not None:
            exp_vec[index] += 1

    return exp_vec


def sieve_polynomial(
    poly: SIQSPolynomial,
    context: MPQSFactorBaseContext,
    M: int,
    a_primes: Optional[List[int]] = None,
) -> List[Tuple[int, int, List[int]]]:
    """
    Sieve one polynomial over interval [-M, M].
    
    Uses logarithmic sieving: initialize with log|Q(x)|, subtract log(p)
    at positions divisible by p. Small residuals indicate smooth values.
    """
    sieve_size = 2 * M
    sieve_log = np.zeros(sieve_size, dtype=np.float64) if context.use_numba else [0.0] * sieve_size
    a_prime_list = a_primes or []
    
    # Initialize with log|Q(x)/A|
    for i in range(sieve_size):
        x = i - M
        _, Q_div_A = poly.evaluate_divided(x)
        value = abs(Q_div_A)
        sieve_log[i] = math.log(value) if value else 0.0
    
    # Sieve with each prime
    if context.use_numba:
        starts: List[Tuple[int, int]] = []
        for p in context.positive_base:
            roots = poly.roots.get(p)
            if roots is None:
                starts.append((-1, -1))
                continue
            root1, root2 = roots
            starts.append(((root1 + M) % p, (root2 + M) % p))
        sieve_with_numba(sieve_log, context.positive_base, starts, poly.n, 0, -M)
    else:
        for p in context.positive_base:
            roots = poly.roots.get(p)
            if roots is None:
                continue

            log_p = context.log_by_prime[p]
            root1, root2 = roots
            start1 = (root1 + M) % p
            start2 = (root2 + M) % p

            for i in range(start1, sieve_size, p):
                sieve_log[i] -= log_p

            if start2 != start1:
                for i in range(start2, sieve_size, p):
                    sieve_log[i] -= log_p
    
    relations = []
    for i in range(sieve_size):
        if sieve_log[i] < context.threshold:
            x = i - M
            Ax_B, Q_div_A = poly.evaluate_divided(x)
            
            if Q_div_A == 0:
                continue

            exp_vec = _factor_relation_value(Q_div_A, a_prime_list, context)
            if exp_vec is not None:
                relations.append((Ax_B, abs(Q_div_A) * poly.A, exp_vec))
    
    return relations


def _resolve_mpqs_workers(n: int, num_workers: Optional[int]) -> int:
    """Choose a worker count that avoids multiprocessing overhead on small inputs."""
    if num_workers is not None:
        return max(1, num_workers)

    if len(str(n)) < 25:
        return 1

    return max(1, os.cpu_count() or 1)


def _select_siqs_polynomial(
    factor_base: List[int],
    target_A: int,
    num_A_primes: int,
    used_A: Set[int],
) -> Optional[Tuple[int, List[int]]]:
    """Find an unused A = product(A_primes), falling back to random choices when needed."""
    candidates = [p for p in factor_base if p > 10]
    max_attempts = max(16, num_A_primes * 8)

    for attempt in range(max_attempts):
        if attempt == 0:
            A_primes = select_A_primes(factor_base, target_A, num_A_primes)
        elif len(candidates) >= num_A_primes:
            A_primes = random.sample(candidates, num_A_primes)
        else:
            A_primes = None

        if A_primes is None:
            return None

        A = math.prod(A_primes)
        if A not in used_A:
            return A, A_primes

    return None


def _build_siqs_batch(
    n: int,
    context: MPQSFactorBaseContext,
    target_A: int,
    num_A_primes: int,
    used_A: Set[int],
    M: int,
    max_polynomials: int,
) -> List[Tuple[int, int, int, MPQSFactorBaseContext, int, List[int]]]:
    """Build a batch of unique polynomials for serial or parallel sieving."""
    polynomial_params: List[Tuple[int, int, int, MPQSFactorBaseContext, int, List[int]]] = []

    while len(polynomial_params) < max_polynomials:
        selected = _select_siqs_polynomial(context.factor_base, target_A, num_A_primes, used_A)
        if selected is None:
            break

        A, A_primes = selected
        B_values = compute_B_values(A_primes, n, context.sqrt_n_mod_p)
        if not B_values:
            continue

        used_A.add(A)

        for B in B_values:
            polynomial_params.append((A, B, n, context, M, A_primes))
            if len(polynomial_params) >= max_polynomials:
                break

    return polynomial_params


def _collect_siqs_relations(
    n: int,
    time_limit: float = 300,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Self-Initializing Quadratic Sieve - collect relations and factor base.
    """
    import time
    start_time = time.time()
    
    # Get optimal parameters
    fb_size, M, max_polys = get_mpqs_params(n)
    context = _build_factor_base_context(n, fb_size)
    target_relations = len(context.factor_base) + 20
    workers = _resolve_mpqs_workers(n, num_workers)
    
    if verbose:
        digits = len(str(n))
        print(
            f"SIQS: {digits} digits, FB={len(context.factor_base)}, "
            f"M={M}, target={target_relations}, workers={workers}"
        )
    
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

        remaining_polys = max_polys - poly_count
        batch_size = 1 if workers == 1 else min(remaining_polys, max(workers * 4, 8))
        polynomial_params = _build_siqs_batch(
            n,
            context,
            target_A,
            num_A_primes,
            used_A,
            M,
            batch_size,
        )
        if not polynomial_params:
            break

        if workers > 1 and len(polynomial_params) > 1:
            from .parallel import parallel_sieve

            new_rels = parallel_sieve(polynomial_params, num_workers=workers)
        else:
            new_rels = []
            for A, B, _, poly_context, _, a_primes in polynomial_params:
                poly = SIQSPolynomial(A, B, n, poly_context.factor_base, poly_context.sqrt_n_mod_p)
                new_rels.extend(sieve_polynomial(poly, poly_context, M, a_primes))

        relations.extend(new_rels)
        poly_count += len(polynomial_params)

        if verbose and poly_count % 50 == 0:
            print_progress(len(relations), target_relations, f"SIQS ({poly_count} polys)")
    
    if verbose:
        print()
    
    return relations, context.positive_base


def siqs(
    n: int,
    time_limit: float = 300,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Self-Initializing Quadratic Sieve - main relation collection."""
    relations, _ = _collect_siqs_relations(n, time_limit, verbose, num_workers)
    return relations


def mpqs_factor(
    n: int,
    time_limit: float = 300,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Factor n using MPQS/SIQS.
    
    Args:
        n: Number to factor (works best for 20-60 digit semiprimes)
        time_limit: Maximum time in seconds
        verbose: Print progress
        num_workers: Worker processes for the sieving phase. Defaults to
            auto-selection based on input size.
    
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
    relations, factor_base = _collect_siqs_relations(
        n,
        time_limit * 0.8,
        verbose,
        num_workers=num_workers,
    )
    
    if verbose:
        print(f"Collected {len(relations)} relations")
    
    if len(relations) < len(factor_base) // 2:
        if verbose:
            print("Insufficient relations")
        return (n, 1)
    
    # Try random combinations first
    if verbose:
        print("Searching for factor...")

    factored_relations = prefactor_relations(relations, factor_base)
    factor = random_combination_search(factored_relations, factor_base, n, max_attempts=2000)
    if factor:
        return (factor, n // factor)
    
    # Full linear algebra
    if verbose:
        print("Running linear algebra...")
    
    dependencies = find_dependencies(factored_relations, factor_base)
    
    if not dependencies:
        if verbose:
            print("No dependencies found")
        return (n, 1)
    
    # Extract factor
    for dep in dependencies[:50]:
        x_prod = 1
        total_exp = [0] * len(factor_base)
        
        for i, bit in enumerate(dep):
            if bit and i < len(factored_relations):
                x_val, _, exp_vec = factored_relations[i]
                x_prod = (x_prod * x_val) % n

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

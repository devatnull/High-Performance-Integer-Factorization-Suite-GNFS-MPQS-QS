"""
Parallel Processing for Factorization Algorithms.

Provides parallelized versions of sieving algorithms using multiprocessing.
The sieving phase of QS/MPQS/GNFS is embarrassingly parallel - each
polynomial can be sieved independently.

Usage:
    from factorization.parallel import parallel_mpqs_factor
    p, q = parallel_mpqs_factor(n, num_workers=4)

Note: For small numbers, the overhead of multiprocessing exceeds the
benefit. Parallelization helps for numbers > 40 digits.
"""

import multiprocessing as mp
import os
from typing import List, Tuple, Optional, Callable

from .utils import gcd, isqrt, is_prime, generate_primes, factor_over_base
from .linear_algebra import find_dependencies, random_combination_search


def _preferred_context() -> mp.context.BaseContext:
    """
    Pick the fastest safe multiprocessing context for the current platform.

    On POSIX, ``fork`` avoids the heavy spawn/pickle startup cost that dominates
    MPQS batches. Windows only supports spawn-style semantics, so we fall back
    to the default context there.
    """
    if os.name != "nt":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context()


def _resolve_worker_count(num_workers: Optional[int], task_count: int) -> int:
    """Clamp the worker count to the amount of useful work."""
    if task_count <= 0:
        return 1

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    return max(1, min(num_workers, task_count))


def _map_in_pool(worker: Callable, tasks: List[Tuple], num_workers: Optional[int]) -> List:
    """Run a top-level worker across tasks with sane chunking."""
    workers = _resolve_worker_count(num_workers, len(tasks))
    if workers == 1:
        return [worker(task) for task in tasks]

    ctx = _preferred_context()
    chunksize = max(1, len(tasks) // (workers * 4))
    with ctx.Pool(workers) as pool:
        return pool.map(worker, tasks, chunksize=chunksize)


def _sieve_polynomial_worker(args: Tuple) -> List[Tuple[int, int]]:
    """
    Worker function for parallel sieving.
    
    Takes polynomial parameters, performs sieving, returns relations.
    Must be a top-level function for pickling.
    """
    from .mpqs import SIQSPolynomial, sieve_polynomial
    
    A, B, n, context, M, a_primes = args
    
    try:
        poly = SIQSPolynomial(A, B, n, context.factor_base, context.sqrt_n_mod_p)
        relations = sieve_polynomial(poly, context, M, a_primes)
        return relations
    except Exception:
        return []


def _ecm_worker(args: Tuple[int, int, int]) -> Optional[int]:
    """Top-level ECM worker for spawn/fork compatibility."""
    from .ecm import ecm_one_curve

    n, B1, B2 = args
    return ecm_one_curve(n, B1, B2)


def _trial_worker(args: Tuple[int, List[int]]) -> Optional[int]:
    """Top-level trial-division worker for spawn/fork compatibility."""
    n, prime_chunk = args
    for p in prime_chunk:
        if n % p == 0:
            return p
    return None


def parallel_sieve(polynomial_params: List[Tuple], 
                   num_workers: int = None) -> List[Tuple[int, int]]:
    """
    Parallel sieving across multiple polynomials.
    
    Args:
        polynomial_params: List of (A, B, n, factor_base, sqrt_n_mod_p, M)
        num_workers: Number of worker processes (default: CPU count)
    
    Returns:
        Combined list of relations from all polynomials
    """
    num_workers = _resolve_worker_count(num_workers, len(polynomial_params))
    
    # For small batches, don't bother with parallelism
    if len(polynomial_params) < num_workers * 2:
        results = []
        for params in polynomial_params:
            results.extend(_sieve_polynomial_worker(params))
        return results
    
    # Parallel execution
    results = _map_in_pool(_sieve_polynomial_worker, polynomial_params, num_workers)
    
    # Flatten results
    all_relations = []
    for rel_list in results:
        all_relations.extend(rel_list)
    
    return all_relations


def parallel_mpqs_factor(n: int, num_workers: int = None, 
                         time_limit: float = 300,
                         verbose: bool = True) -> Tuple[int, int]:
    """
    Parallel MPQS factorization.
    
    Distributes polynomial sieving across multiple CPU cores.
    
    Args:
        n: Number to factor
        num_workers: Number of worker processes
        time_limit: Maximum time in seconds
        verbose: Print progress
    
    Returns:
        (p, q) where p * q = n, or (n, 1) if failed
    """
    import time
    from .mpqs import (
        _build_factor_base_context,
        compute_B_values,
        get_mpqs_params,
        select_A_primes,
    )
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    start_time = time.time()
    
    if n <= 1:
        return (n, 1)
    if is_prime(n):
        return (n, 1)
    
    # Quick trial division
    for p in generate_primes(100000):
        if n % p == 0:
            return (p, n // p)
    
    # Get parameters
    fb_size, M, max_polys = get_mpqs_params(n)
    context = _build_factor_base_context(n, fb_size)
    target_relations = len(context.factor_base) + 20
    
    if verbose:
        digits = len(str(n))
        print(f"Parallel MPQS: {digits} digits, {num_workers} workers")
        print(f"Factor base: {len(context.factor_base)}, target: {target_relations}")
    
    # Generate polynomial parameters
    sqrt_2n = isqrt(2 * n)
    target_A = sqrt_2n // M
    num_A_primes = 3 if len(str(n)) < 30 else 4
    
    used_A = set()
    poly_params = []
    
    # Prepare batch of polynomials
    batch_size = num_workers * 10
    
    while len(poly_params) < batch_size:
        A_primes = select_A_primes(context.factor_base, target_A, num_A_primes)
        if A_primes is None:
            break
        
        A = 1
        for p in A_primes:
            A *= p
        
        if A in used_A:
            continue
        used_A.add(A)
        
        B_values = compute_B_values(A_primes, n, context.sqrt_n_mod_p)
        for B in B_values:
            poly_params.append((A, B, n, context, M, A_primes))
    
    # Parallel sieving
    if verbose:
        print(f"Sieving {len(poly_params)} polynomials...")
    
    relations = parallel_sieve(poly_params, num_workers)
    
    if verbose:
        print(f"Collected {len(relations)} relations")
    
    # Factor base for linear algebra (remove -1)
    fb_positive = context.positive_base
    
    if len(relations) < len(fb_positive) // 2:
        if verbose:
            print("Insufficient relations")
        return (n, 1)
    
    # Try random combinations
    factor = random_combination_search(relations, fb_positive, n, max_attempts=2000)
    if factor:
        return (factor, n // factor)
    
    # Full linear algebra
    if verbose:
        print("Running linear algebra...")
    
    dependencies = find_dependencies(relations, fb_positive)
    
    if not dependencies:
        return (n, 1)
    
    # Extract factor
    for dep in dependencies[:50]:
        x_prod = 1
        total_exp = [0] * len(fb_positive)
        
        for i, bit in enumerate(dep):
            if bit and i < len(relations):
                x_val, y_sq = relations[i]
                x_prod = (x_prod * x_val) % n
                
                exp_vec = factor_over_base(y_sq, fb_positive)
                if exp_vec:
                    for j, e in enumerate(exp_vec):
                        total_exp[j] += e
        
        if not all(e % 2 == 0 for e in total_exp):
            continue
        
        y = 1
        for j, e in enumerate(total_exp):
            y = (y * pow(fb_positive[j], e // 2, n)) % n
        
        for candidate in [x_prod - y, x_prod + y]:
            factor = gcd(candidate, n)
            if 1 < factor < n:
                return (factor, n // factor)
    
    return (n, 1)


def parallel_ecm(n: int, num_workers: int = None, 
                 B1: int = 50000, B2: int = 5000000,
                 curves_per_worker: int = 20) -> Optional[int]:
    """
    Parallel ECM - run multiple curves simultaneously.
    
    Each worker tries different random curves independently.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)
    
    total_curves = num_workers * curves_per_worker
    args_list = [(n, B1, B2) for _ in range(total_curves)]

    ctx = _preferred_context()
    with ctx.Pool(num_workers) as pool:
        for result in pool.imap_unordered(_ecm_worker, args_list):
            if result is not None:
                pool.terminate()
                return result
    
    return None


def parallel_trial_division(n: int, limit: int, num_workers: int = None) -> Optional[int]:
    """
    Parallel trial division - split prime range across workers.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)
    
    primes = generate_primes(limit)
    chunk_size = len(primes) // num_workers + 1

    chunks = [(n, primes[i:i+chunk_size]) for i in range(0, len(primes), chunk_size)]
    num_workers = _resolve_worker_count(num_workers, len(chunks))

    ctx = _preferred_context()
    with ctx.Pool(num_workers) as pool:
        for result in pool.imap_unordered(_trial_worker, chunks):
            if result is not None:
                pool.terminate()
                return result
    
    return None

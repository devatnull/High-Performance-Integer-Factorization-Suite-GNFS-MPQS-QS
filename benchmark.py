#!/usr/bin/env python3
"""
Benchmark suite for integer factorization algorithms.

Compares performance of different algorithms across various number sizes.
Generates timing data and success rates.

Usage:
    python benchmark.py           # Run default benchmark
    python benchmark.py --quick   # Quick benchmark (fewer iterations)
    python benchmark.py --full    # Full benchmark (more iterations)
"""

import time
import random
import argparse
import json
import os
import subprocess
import sys
from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass

from factorization import (
    factorize, trial_division, pollard_rho, ecm, 
    qs_factor, mpqs_factor, is_prime, numba_available, gmpy2_available
)
from factorization.utils import generate_primes


@dataclass
class BenchmarkResult:
    algorithm: str
    n_digits: int
    successes: int
    failures: int
    total_time: float
    avg_time: float
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0


def generate_semiprime(digits: int) -> Tuple[int, int, int]:
    """Generate a semiprime n = p * q with approximately `digits` digits."""
    half_digits = digits // 2
    
    # Generate primes of appropriate size
    lower = 10 ** (half_digits - 1)
    upper = 10 ** half_digits
    
    primes = [p for p in generate_primes(min(upper, 10**7)) if p >= lower]
    
    if not primes:
        # Generate random primes for larger sizes
        while True:
            p = random.randint(lower, upper)
            if is_prime(p):
                break
        while True:
            q = random.randint(lower, upper)
            if is_prime(q) and q != p:
                break
    else:
        p = random.choice(primes)
        q = random.choice([x for x in primes if x != p])
    
    return p * q, p, q


def benchmark_algorithm(algo_func: Callable, algo_name: str,
                       test_numbers: List[Tuple[int, int, int]],
                       time_limit: float = 30) -> BenchmarkResult:
    """
    Benchmark a single algorithm on test numbers.
    
    Args:
        algo_func: Function that takes (n, time_limit) and returns (p, q)
        algo_name: Name for display
        test_numbers: List of (n, expected_p, expected_q) tuples
        time_limit: Max time per factorization
    
    Returns:
        BenchmarkResult with timing and success data
    """
    successes = 0
    failures = 0
    total_time = 0
    
    for n, expected_p, expected_q in test_numbers:
        start = time.time()
        try:
            if algo_name in ["Trial Division"]:
                p, q = algo_func(n)
            elif algo_name in ["Pollard Rho", "ECM"]:
                factor = algo_func(n)
                if factor and factor != n and n % factor == 0:
                    p, q = factor, n // factor
                else:
                    p, q = n, 1
            else:
                p, q = algo_func(n, time_limit=time_limit, verbose=False)
        except Exception:
            p, q = n, 1
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # Check if factorization is correct
        if p * q == n and p != n and p != 1:
            successes += 1
        else:
            failures += 1
    
    avg_time = total_time / len(test_numbers) if test_numbers else 0
    digits = len(str(test_numbers[0][0])) if test_numbers else 0
    
    return BenchmarkResult(
        algorithm=algo_name,
        n_digits=digits,
        successes=successes,
        failures=failures,
        total_time=total_time,
        avg_time=avg_time
    )


def benchmark_mpqs_workers(test_numbers: List[Tuple[int, int, int]], time_limit: float) -> List[BenchmarkResult]:
    """Compare MPQS worker policies on the same input set."""
    worker_modes = [
        (None, "MPQS (adaptive)"),
        (1, "MPQS (1 worker)"),
        (min(4, os.cpu_count() or 1), "MPQS (explicit workers)"),
    ]

    results: List[BenchmarkResult] = []
    for workers, label in worker_modes:
        successes = 0
        failures = 0
        total_time = 0.0

        for n, _, _ in test_numbers:
            start = time.time()
            try:
                p, q = mpqs_factor(n, time_limit=time_limit, verbose=False, num_workers=workers)
            except Exception:
                p, q = n, 1
            total_time += time.time() - start

            if p * q == n and p != n and p != 1:
                successes += 1
            else:
                failures += 1

        results.append(BenchmarkResult(
            algorithm=label,
            n_digits=len(str(test_numbers[0][0])) if test_numbers else 0,
            successes=successes,
            failures=failures,
            total_time=total_time,
            avg_time=total_time / len(test_numbers) if test_numbers else 0.0,
        ))

    return results


def benchmark_mpqs_backend_modes(test_numbers: List[Tuple[int, int, int]], time_limit: float) -> List[dict]:
    """Benchmark MPQS across backend toggles in isolated subprocesses."""
    modes = [
        ("Current env", {}),
        ("No Numba", {"FACTOR_DISABLE_NUMBA": "1"}),
        ("No gmpy2", {"FACTOR_DISABLE_GMPY2": "1"}),
        ("Pure Python", {"FACTOR_DISABLE_NUMBA": "1", "FACTOR_DISABLE_GMPY2": "1"}),
    ]
    payload_numbers = [n for n, _, _ in test_numbers]
    code = (
        "import json, time\n"
        "from factorization import mpqs_factor, numba_available, gmpy2_available\n"
        f"numbers = {payload_numbers!r}\n"
        f"time_limit = {time_limit!r}\n"
        "successes = 0\n"
        "failures = 0\n"
        "total_time = 0.0\n"
        "for n in numbers:\n"
        "    start = time.time()\n"
        "    try:\n"
        "        p, q = mpqs_factor(n, time_limit=time_limit, verbose=False)\n"
        "    except Exception:\n"
        "        p, q = n, 1\n"
        "    total_time += time.time() - start\n"
        "    if p * q == n and p != n and p != 1:\n"
        "        successes += 1\n"
        "    else:\n"
        "        failures += 1\n"
        "print(json.dumps({\n"
        "    'numba': numba_available(),\n"
        "    'gmpy2': gmpy2_available(),\n"
        "    'successes': successes,\n"
        "    'failures': failures,\n"
        "    'avg_time': total_time / len(numbers) if numbers else 0.0,\n"
        "}))\n"
    )

    results = []
    for label, overrides in modes:
        env = os.environ.copy()
        env.update(overrides)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip())
        payload["label"] = label
        results.append(payload)

    return results


def run_benchmark(quick: bool = False, full: bool = False):
    """Run the complete benchmark suite."""
    
    print("=" * 70)
    print("Integer Factorization Benchmark Suite")
    print("=" * 70)
    print(f"Numba JIT: {'Available ✓' if numba_available() else 'Not available'}")
    print(f"gmpy2: {'Available ✓' if gmpy2_available() else 'Not available'}")
    print()
    
    # Define test sizes
    if quick:
        digit_sizes = [8, 12, 16, 20]
        nums_per_size = 3
        time_limit = 10
    elif full:
        digit_sizes = [6, 8, 10, 12, 14, 16, 18, 20, 24]
        nums_per_size = 10
        time_limit = 60
    else:
        digit_sizes = [8, 10, 12, 14, 16, 20]
        nums_per_size = 5
        time_limit = 30
    
    # Generate test numbers
    test_sets: Dict[int, List[Tuple[int, int, int]]] = {}
    for digits in digit_sizes:
        test_sets[digits] = [generate_semiprime(digits) for _ in range(nums_per_size)]
    
    # Define algorithms to test
    algorithms = [
        (trial_division, "Trial Division"),
        (pollard_rho, "Pollard Rho"),
        (ecm, "ECM"),
        (mpqs_factor, "MPQS"),
        (factorize, "Auto (factorize)"),
    ]
    
    # Run benchmarks
    results: List[BenchmarkResult] = []
    
    for digits in digit_sizes:
        print(f"\n{'='*70}")
        print(f"Testing {digits}-digit semiprimes ({nums_per_size} numbers)")
        print(f"{'='*70}")
        
        for algo_func, algo_name in algorithms:
            # Skip slow algorithms for large numbers
            if algo_name == "Trial Division" and digits > 12:
                continue
            if algo_name == "MPQS" and digits < 10:
                continue
            
            print(f"  {algo_name}...", end=" ", flush=True)
            result = benchmark_algorithm(
                algo_func, algo_name, test_sets[digits], time_limit
            )
            results.append(result)
            
            status = "✓" if result.success_rate == 1.0 else f"{result.success_rate:.0%}"
            print(f"{result.avg_time:.3f}s avg, {status}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Digits':>8} {'Success':>10} {'Avg Time':>12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result.algorithm:<20} {result.n_digits:>8} "
              f"{result.success_rate:>9.0%} {result.avg_time:>11.3f}s")

    print("=" * 70)

    mpqs_target_digits = max(d for d in digit_sizes if d >= 12)
    mpqs_numbers = test_sets[mpqs_target_digits]

    print("\n" + "=" * 70)
    print(f"MPQS WORKER COMPARISON ({mpqs_target_digits}-digit set)")
    print("=" * 70)
    worker_results = benchmark_mpqs_workers(mpqs_numbers, time_limit)
    for result in worker_results:
        status = "✓" if result.success_rate == 1.0 else f"{result.success_rate:.0%}"
        print(f"{result.algorithm:<24} {result.avg_time:.3f}s avg, {status}")

    print("\n" + "=" * 70)
    print(f"MPQS BACKEND COMPARISON ({mpqs_target_digits}-digit set)")
    print("=" * 70)
    for result in benchmark_mpqs_backend_modes(mpqs_numbers, time_limit):
        status = "✓" if result["failures"] == 0 else f"{result['failures']} fail"
        print(
            f"{result['label']:<14} avg={result['avg_time']:.3f}s, "
            f"numba={'on' if result['numba'] else 'off'}, "
            f"gmpy2={'on' if result['gmpy2'] else 'off'}, {status}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark factorization algorithms")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--full", action="store_true", help="Full benchmark")
    args = parser.parse_args()
    
    run_benchmark(quick=args.quick, full=args.full)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Command-line interface for integer factorization.

Usage:
    python -m factorization 1234567
    python -m factorization 1234567 --algorithm ecm
    python -m factorization 1234567 --verbose --timeout 60
    echo "1234567" | python -m factorization -
"""

import argparse
import sys
import time
from typing import Optional

from . import (
    factorize, factorize_full, is_prime,
    trial_division, pollard_rho, pollard_pm1, williams_pp1,
    ecm, qs_factor, mpqs_factor, gnfs_factor, numba_available
)
from .fermat import fermat_factor
from .squfof import squfof_factor


ALGORITHMS = {
    'auto': factorize,
    'trial': lambda n, **kw: trial_division(n),
    'fermat': lambda n, **kw: fermat_factor(n),
    'rho': lambda n, **kw: (pollard_rho(n) or n, n // (pollard_rho(n) or n) or 1),
    'squfof': lambda n, **kw: squfof_factor(n),
    'pm1': lambda n, **kw: (pollard_pm1(n) or n, n // (pollard_pm1(n) or n) or 1),
    'pp1': lambda n, **kw: (williams_pp1(n) or n, n // (williams_pp1(n) or n) or 1),
    'ecm': lambda n, **kw: (ecm(n) or n, n // (ecm(n) or n) or 1),
    'qs': qs_factor,
    'mpqs': mpqs_factor,
    'gnfs': gnfs_factor,
}


def format_factorization(factors: dict) -> str:
    """Format factorization as string."""
    if not factors:
        return "1"
    
    parts = []
    for prime in sorted(factors.keys()):
        exp = factors[prime]
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    
    return " × ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="High-performance integer factorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:
  auto    Automatic selection (default)
  trial   Trial division - O(√n)
  fermat  Fermat's method - good for close factors
  rho     Pollard's Rho - O(n^1/4)
  squfof  Shanks' SQUFOF - O(n^1/4)
  pm1     Pollard p-1 - for smooth (p-1)
  pp1     Williams p+1 - for smooth (p+1)
  ecm     Elliptic Curve Method
  qs      Quadratic Sieve
  mpqs    Multiple Polynomial QS
  gnfs    General Number Field Sieve

Examples:
  python -m factorization 1234567
  python -m factorization 99999999999999997 --algorithm ecm
  python -m factorization 12345678901234567890 --full --verbose
""")
    
    parser.add_argument('number', nargs='?', default=None,
                        help="Number to factor (or - for stdin)")
    parser.add_argument('-a', '--algorithm', choices=list(ALGORITHMS.keys()),
                        default='auto', help="Factorization algorithm")
    parser.add_argument('-f', '--full', action='store_true',
                        help="Complete factorization into prime powers")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output")
    parser.add_argument('-t', '--timeout', type=float, default=300,
                        help="Timeout in seconds (default: 300)")
    parser.add_argument('--info', action='store_true',
                        help="Show system info and exit")
    
    args = parser.parse_args()
    
    # Show info
    if args.info:
        print("Integer Factorization Suite")
        print(f"  Numba JIT: {'Available' if numba_available() else 'Not available'}")
        print(f"  Algorithms: {', '.join(ALGORITHMS.keys())}")
        try:
            import gmpy2
            print("  gmpy2: Available")
        except ImportError:
            print("  gmpy2: Not available")
        return 0
    
    # Get number
    if args.number is None:
        parser.print_help()
        return 1
    
    if args.number == '-':
        # Read from stdin
        try:
            n_str = sys.stdin.read().strip()
        except KeyboardInterrupt:
            return 1
    else:
        n_str = args.number
    
    try:
        n = int(n_str)
    except ValueError:
        print(f"Error: '{n_str}' is not a valid integer", file=sys.stderr)
        return 1
    
    if n <= 0:
        print(f"Error: number must be positive", file=sys.stderr)
        return 1
    
    # Basic info
    digits = len(str(n))
    if args.verbose:
        print(f"Input: {n} ({digits} digits)")
        print(f"Algorithm: {args.algorithm}")
        if is_prime(n):
            print(f"Result: {n} is prime")
            return 0
        print()
    
    # Factor
    start_time = time.time()
    
    try:
        if args.full:
            factors = factorize_full(n, time_limit=args.timeout, verbose=args.verbose)
            elapsed = time.time() - start_time
            
            # Output
            result_str = format_factorization(factors)
            print(f"{n} = {result_str}")
            
        else:
            algo_func = ALGORITHMS[args.algorithm]
            
            if args.algorithm == 'auto':
                p, q = algo_func(n, time_limit=args.timeout, verbose=args.verbose)
            elif args.algorithm in ['qs', 'mpqs', 'gnfs']:
                p, q = algo_func(n, time_limit=args.timeout, verbose=args.verbose)
            else:
                p, q = algo_func(n, time_limit=args.timeout)
            
            elapsed = time.time() - start_time
            
            if p == n or q == 1:
                if is_prime(n):
                    print(f"{n} is prime")
                else:
                    print(f"Factorization failed for {n}")
                    return 1
            else:
                # Ensure p <= q
                if p > q:
                    p, q = q, p
                print(f"{n} = {p} × {q}")
        
        if args.verbose:
            print(f"\nTime: {elapsed:.3f}s")
    
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Interactive TUI for Integer Factorization Suite.

Run with: python -m factorization.tui
"""

import sys
import time
import os
from typing import Optional, Tuple, Dict, Callable

# Import all algorithms
from . import (
    factorize, factorize_full, is_prime,
    trial_division, pollard_rho, pollard_pm1, williams_pp1,
    ecm, qs_factor, mpqs_factor, gnfs_factor,
    estimate_gnfs_runtime, numba_available
)
from .fermat import fermat_factor
from .squfof import squfof_factor


BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   INTEGER FACTORIZATION SUITE v1.0                            ║
║                                                                               ║
║   Algorithms: Trial Division, Fermat, Pollard Rho, SQUFOF, p-1, p+1,         ║
║               ECM, Quadratic Sieve, MPQS, GNFS                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

MENU = """
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMMANDS                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [f] Factor a number (auto-select algorithm)                                │
│  [a] Factor with specific algorithm                                         │
│  [b] Run benchmark                                                          │
│  [e] Estimate GNFS resources for a number                                   │
│  [p] Check if number is prime                                               │
│  [i] System info                                                            │
│  [h] Help                                                                   │
│  [q] Quit                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
"""

ALGORITHMS = {
    '1': ('Trial Division', lambda n, **kw: trial_division(n), 'O(√n), best for n < 10^12'),
    '2': ('Fermat', lambda n, **kw: fermat_factor(n), 'Good when factors are close'),
    '3': ('Pollard Rho', lambda n, **kw: _wrap_single(pollard_rho, n), 'O(n^1/4), general purpose'),
    '4': ('SQUFOF', lambda n, **kw: squfof_factor(n), 'O(n^1/4), low memory'),
    '5': ('Pollard p-1', lambda n, **kw: _wrap_single(pollard_pm1, n), 'For smooth (p-1)'),
    '6': ('Williams p+1', lambda n, **kw: _wrap_single(williams_pp1, n), 'For smooth (p+1)'),
    '7': ('ECM', lambda n, **kw: _wrap_single(lambda x: ecm(x, B1=50000, max_curves=100), n), 'Best for medium factors'),
    '8': ('Quadratic Sieve', lambda n, **kw: qs_factor(n, **kw), 'L[1/2,1], 25-50 digits'),
    '9': ('MPQS', lambda n, **kw: mpqs_factor(n, **kw), 'L[1/2,1], 25-80 digits'),
    '0': ('GNFS', lambda n, **kw: gnfs_factor(n, **kw), 'L[1/3,1.9], 50+ digits'),
}


def _wrap_single(func, n) -> Tuple[int, int]:
    """Wrap single-factor-returning functions."""
    result = func(n)
    if result and result != n:
        return (result, n // result)
    return (n, 1)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_colored(text: str, color: str = 'white'):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def get_number(prompt: str = "Enter number: ") -> Optional[int]:
    try:
        text = input(prompt).strip()
        if not text:
            return None
        # Handle scientific notation
        if 'e' in text.lower() or '^' in text:
            text = text.replace('^', '**').replace('x', '*')
            return int(eval(text))
        return int(text)
    except (ValueError, SyntaxError):
        print_colored("Invalid number format", 'red')
        return None
    except KeyboardInterrupt:
        return None


def factor_auto():
    """Factor using automatic algorithm selection."""
    print_colored("\n═══ AUTO FACTORIZATION ═══", 'cyan')
    n = get_number()
    if n is None or n < 2:
        return
    
    digits = len(str(n))
    print(f"\nFactoring {n} ({digits} digits)...")
    print_colored("Algorithm will be auto-selected based on size\n", 'yellow')
    
    start = time.time()
    try:
        p, q = factorize(n, verbose=True)
        elapsed = time.time() - start
        
        print()
        if p != n and q != 1:
            print_colored(f"✓ {n} = {p} × {q}", 'green')
            
            # Show full factorization if composite factors
            if not is_prime(p) or not is_prime(q):
                print_colored("\nFull prime factorization:", 'yellow')
                factors = factorize_full(n)
                parts = [f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items())]
                print_colored(f"  {n} = {' × '.join(parts)}", 'green')
        else:
            if is_prime(n):
                print_colored(f"✓ {n} is PRIME", 'green')
            else:
                print_colored(f"✗ Factorization failed", 'red')
        
        print(f"\nTime: {elapsed:.3f}s")
    except KeyboardInterrupt:
        print_colored("\nCancelled", 'yellow')


def factor_specific():
    """Factor using a specific algorithm."""
    print_colored("\n═══ SELECT ALGORITHM ═══", 'cyan')
    print()
    for key, (name, _, desc) in ALGORITHMS.items():
        print(f"  [{key}] {name:<20} - {desc}")
    print()
    
    choice = input("Select algorithm (1-0): ").strip()
    if choice not in ALGORITHMS:
        print_colored("Invalid choice", 'red')
        return
    
    name, func, _ = ALGORITHMS[choice]
    print_colored(f"\nUsing: {name}", 'yellow')
    
    n = get_number()
    if n is None or n < 2:
        return
    
    digits = len(str(n))
    print(f"\nFactoring {n} ({digits} digits)...")
    
    start = time.time()
    try:
        p, q = func(n, time_limit=300, verbose=True)
        elapsed = time.time() - start
        
        print()
        if p != n and q != 1:
            p, q = min(p, q), max(p, q)
            print_colored(f"✓ {n} = {p} × {q}", 'green')
        else:
            if is_prime(n):
                print_colored(f"✓ {n} is PRIME", 'green')
            else:
                print_colored(f"✗ Algorithm failed to find factors", 'red')
        
        print(f"\nTime: {elapsed:.3f}s")
    except KeyboardInterrupt:
        print_colored("\nCancelled", 'yellow')
    except Exception as e:
        print_colored(f"\nError: {e}", 'red')


def run_benchmark():
    """Run benchmark on various algorithms."""
    print_colored("\n═══ BENCHMARK ═══", 'cyan')
    print()
    print("  [1] Quick (8-16 digit semiprimes)")
    print("  [2] Standard (8-24 digit semiprimes)")  
    print("  [3] Extended (8-32 digit semiprimes)")
    print()
    
    choice = input("Select benchmark level (1-3): ").strip()
    
    if choice == '1':
        sizes = [(8, 3), (12, 3), (16, 3)]
    elif choice == '2':
        sizes = [(8, 3), (12, 3), (16, 3), (20, 2), (24, 2)]
    elif choice == '3':
        sizes = [(8, 3), (12, 3), (16, 3), (20, 2), (24, 2), (28, 2), (32, 1)]
    else:
        print_colored("Invalid choice", 'red')
        return
    
    import random
    
    def gen_semiprime(digits):
        half = digits // 2
        lower = 10 ** (half - 1)
        upper = 10 ** half
        while True:
            p = random.randint(lower, upper)
            q = random.randint(lower, upper)
            if is_prime(p) and is_prime(q):
                return p * q
    
    print_colored("\nRunning benchmark...\n", 'yellow')
    
    algos = [
        ('Auto', factorize),
        ('Pollard Rho', lambda n, **kw: _wrap_single(pollard_rho, n)),
        ('ECM', lambda n, **kw: _wrap_single(lambda x: ecm(x, B1=10000, max_curves=50), n)),
    ]
    
    results = {name: [] for name, _ in algos}
    
    for digit_count, num_tests in sizes:
        print(f"Testing {digit_count}-digit semiprimes ({num_tests} numbers):")
        
        test_numbers = [gen_semiprime(digit_count) for _ in range(num_tests)]
        
        for algo_name, algo_func in algos:
            successes = 0
            total_time = 0
            
            for n in test_numbers:
                try:
                    start = time.time()
                    p, q = algo_func(n, time_limit=30, verbose=False)
                    elapsed = time.time() - start
                    
                    if p * q == n and p != n:
                        successes += 1
                        total_time += elapsed
                except:
                    pass
            
            rate = (successes / num_tests) * 100
            avg_time = total_time / max(successes, 1)
            results[algo_name].append((digit_count, rate, avg_time))
            
            status = '✓' if rate == 100 else '○' if rate > 0 else '✗'
            print(f"  {status} {algo_name:<15} {rate:5.0f}%  {avg_time:.3f}s avg")
        
        print()
    
    print_colored("Benchmark complete!", 'green')


def estimate_resources():
    """Estimate GNFS resources for a number."""
    print_colored("\n═══ GNFS RESOURCE ESTIMATOR ═══", 'cyan')
    print()
    print("Enter a number or bit size (e.g., '512' for RSA-512):")
    
    text = input("> ").strip()
    
    try:
        if len(text) <= 4 and text.isdigit():
            # Assume bit size
            bits = int(text)
            n = 2 ** bits
            print(f"\nEstimating for {bits}-bit number...")
        else:
            n = int(text)
            bits = n.bit_length()
            print(f"\nEstimating for {len(str(n))}-digit ({bits}-bit) number...")
        
        est = estimate_gnfs_runtime(n)
        
        print()
        print_colored("┌─────────────────────────────────────────┐", 'cyan')
        print_colored("│         GNFS RESOURCE ESTIMATE          │", 'cyan')
        print_colored("├─────────────────────────────────────────┤", 'cyan')
        print(f"│  Digits:        {est['digits']:>20} │")
        print(f"│  Bits:          {est['bits']:>20} │")
        print(f"│  Core-years:    {est['core_years']:>20.2e} │")
        print(f"│  Core-hours:    {est['core_hours']:>20.2e} │")
        print(f"│  Relations:     {est['estimated_relations']:>20.2e} │")
        print(f"│  Memory (GB):   {est['estimated_memory_gb']:>20.2e} │")
        print_colored("└─────────────────────────────────────────┘", 'cyan')
        
        # Context
        print()
        if est['core_years'] < 0.01:
            print_colored("→ Easily factored with current hardware", 'green')
        elif est['core_years'] < 1:
            print_colored("→ Factored in hours to days on modern PC", 'yellow')
        elif est['core_years'] < 1000:
            print_colored("→ Requires significant compute cluster", 'yellow')
        elif est['core_years'] < 1e6:
            print_colored("→ Major research effort (nation-state level)", 'red')
        else:
            print_colored("→ Computationally infeasible with known technology", 'red')
            
    except Exception as e:
        print_colored(f"Error: {e}", 'red')


def check_prime():
    """Check if a number is prime."""
    print_colored("\n═══ PRIMALITY TEST ═══", 'cyan')
    n = get_number()
    if n is None:
        return
    
    start = time.time()
    result = is_prime(n)
    elapsed = time.time() - start
    
    if result:
        print_colored(f"\n✓ {n} is PRIME", 'green')
    else:
        print_colored(f"\n✗ {n} is COMPOSITE", 'yellow')
        # Show smallest factor
        for p in [2, 3, 5, 7, 11, 13]:
            if n % p == 0:
                print(f"  (divisible by {p})")
                break
    
    print(f"\nTime: {elapsed:.6f}s")


def show_info():
    """Show system information."""
    print_colored("\n═══ SYSTEM INFO ═══", 'cyan')
    print()
    print(f"  Python version:  {sys.version.split()[0]}")
    print(f"  Numba JIT:       {'Available ✓' if numba_available() else 'Not available'}")
    
    try:
        import gmpy2
        print(f"  gmpy2:           Available ✓ (fast arithmetic)")
    except ImportError:
        print(f"  gmpy2:           Not available")
    
    try:
        import numpy as np
        print(f"  NumPy:           {np.__version__}")
    except ImportError:
        print(f"  NumPy:           Not available")
    
    print()
    print("  Algorithms available:")
    for key, (name, _, desc) in ALGORITHMS.items():
        print(f"    • {name}")


def show_help():
    """Show help information."""
    print_colored("\n═══ HELP ═══", 'cyan')
    print("""
  This tool provides various integer factorization algorithms.
  
  QUICK START:
    • Press 'f' to factor a number (algorithm auto-selected)
    • Press 'a' to choose a specific algorithm
    • Enter numbers in decimal, or use 2^64 / 10**20 notation
  
  ALGORITHM SELECTION:
    • Small numbers (< 20 digits): Pollard Rho is fastest
    • Medium numbers (20-50 digits): ECM works well
    • Large numbers (50+ digits): MPQS or GNFS
  
  TIPS:
    • For semiprimes (p×q), ECM is often best if one factor is small
    • GNFS is only worthwhile for 80+ digit numbers
    • Use 'e' to estimate resources needed for large numbers
  
  EXAMPLES:
    • Factor RSA-100: Enter the 100-digit RSA challenge number
    • Test primality: Use 'p' command
    • Benchmark: Use 'b' to compare algorithm speeds
""")


def main():
    """Main TUI loop."""
    clear_screen()
    print_colored(BANNER, 'cyan')
    
    while True:
        print(MENU)
        
        try:
            choice = input("Command: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        
        if choice == 'q':
            print_colored("\nGoodbye!", 'cyan')
            break
        elif choice == 'f':
            factor_auto()
        elif choice == 'a':
            factor_specific()
        elif choice == 'b':
            run_benchmark()
        elif choice == 'e':
            estimate_resources()
        elif choice == 'p':
            check_prime()
        elif choice == 'i':
            show_info()
        elif choice == 'h':
            show_help()
        elif choice == '':
            continue
        else:
            print_colored("Unknown command. Press 'h' for help.", 'yellow')
        
        print()
        input("Press Enter to continue...")
        clear_screen()
        print_colored(BANNER, 'cyan')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Legacy entry point - use the factorization package directly instead.

Example:
    from factorization import factorize, factorize_full
    
    p, q = factorize(n)
    factors = factorize_full(n)
    
Or use the CLI:
    python -m factorization 1234567 --verbose
"""

from factorization import factorize, factorize_full, is_prime


def main():
    print("Integer Factorization Suite")
    print("=" * 50)
    print()
    print("Use the factorization package directly:")
    print("  from factorization import factorize")
    print("  p, q = factorize(n)")
    print()
    print("Or use the CLI:")
    print("  python -m factorization 1234567 --verbose")
    print()
    
    # Quick demo
    test_numbers = [
        15,            # 3 × 5
        143,           # 11 × 13
        10403,         # 101 × 103
        1000003 * 1000033,  # 13-digit semiprime
    ]
    
    print("Quick demo:")
    print("-" * 50)
    for n in test_numbers:
        p, q = factorize(n)
        status = "✓" if p * q == n and p > 1 and q > 1 else "✗"
        print(f"{n} = {p} × {q}  {status}")


if __name__ == "__main__":
    main()

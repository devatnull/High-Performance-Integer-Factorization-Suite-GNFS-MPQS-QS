"""
Shanks' Square Forms Factorization (SQUFOF).

One of the most elegant factorization algorithms, using the theory of
binary quadratic forms and continued fractions.

The algorithm operates on the principal form (1, 0, -n) and repeatedly
applies reduction steps. When we encounter a square form (a, b, c) where
a is a perfect square, we can extract a factor.

Key insight: The continued fraction expansion of √n produces a sequence
of quadratic forms. By tracking when we hit a "square form" in the
principal cycle, we can compute gcd to find factors.

Complexity: O(n^1/4) - same as Pollard Rho but often faster in practice
Space: O(1) - only needs a few integers

Best for:
- Numbers up to ~60 bits (before overflow concerns)
- When memory is constrained
- As a fast preliminary check

References:
    - Shanks, D. "Analysis and Improvement of the Continued Fraction Method" (1975)
    - Gower & Wagstaff, "Square Form Factorization" (2008) - modern analysis
    - Cohen, H. "A Course in Computational Algebraic Number Theory" - Ch 5.4
"""

import math
from typing import Optional, Tuple
from .utils import isqrt, gcd, is_prime


def squfof(n: int, max_iterations: int = 1000000) -> Optional[int]:
    """
    Shanks' Square Forms Factorization.
    
    Uses continued fraction expansion of √n to find factors.
    When a square form is encountered in the principal cycle,
    gcd reveals a factor.
    
    Args:
        n: Number to factor (should be odd and not a perfect square)
        max_iterations: Maximum iterations in forward/reverse cycles
    
    Returns:
        A non-trivial factor of n, or None if not found
    
    Example:
        >>> squfof(11111)  # 41 * 271
        41
        >>> squfof(1000003 * 1000033)
        1000003
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    if is_prime(n):
        return None
    
    # Check for perfect square
    sqrt_n = isqrt(n)
    if sqrt_n * sqrt_n == n:
        return sqrt_n
    
    # Try different multipliers for better performance
    multipliers = [1, 3, 5, 7, 11, 3*5, 3*7, 3*11, 5*7, 5*11, 7*11,
                   3*5*7, 3*5*11, 3*7*11, 5*7*11, 3*5*7*11]
    
    for k in multipliers:
        kn = k * n
        
        # Skip if kn is a perfect square
        sqrt_kn = isqrt(kn)
        if sqrt_kn * sqrt_kn == kn:
            continue
        
        factor = squfof_one_multiplier(kn, n, max_iterations // len(multipliers))
        if factor is not None and 1 < factor < n:
            return factor
    
    return None


def squfof_one_multiplier(kn: int, n: int, max_iterations: int) -> Optional[int]:
    """
    SQUFOF with a single multiplier.
    
    The algorithm has two phases:
    1. Forward cycle: iterate until we find a square form
    2. Reverse cycle: continue from square form to find factor
    """
    sqrt_kn = isqrt(kn)
    
    # Initialize forward cycle
    # We track (P, Q) where the form is (Q_{i-1}, 2P_i, Q_i)
    P_prev = 0
    Q_prev = 1
    P = sqrt_kn
    Q = kn - P * P
    
    if Q == 0:
        return None
    
    # Forward cycle: search for square form
    # A form (Q_prev, 2P, Q) is a square form if Q is a perfect square
    
    iteration = 0
    found_square = False
    sqrt_Q = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Compute next (P, Q) using continued fraction recurrence
        # b = floor((sqrt_kn + P) / Q)
        b = (sqrt_kn + P) // Q
        P_new = b * Q - P
        Q_new = Q_prev + b * (P - P_new)
        
        # Check for square form (at even iterations only for proper cycle)
        if iteration % 2 == 0:
            sqrt_Q = isqrt(Q)
            if sqrt_Q * sqrt_Q == Q:
                found_square = True
                break
        
        # Advance
        P_prev, Q_prev = P, Q
        P, Q = P_new, Q_new
        
        if Q == 0:
            return None
    
    if not found_square:
        return None
    
    # Reverse cycle: start from square form and iterate until symmetry
    # Initialize reverse with the square root
    Q_prev = sqrt_Q
    P = P - sqrt_Q * ((P - sqrt_kn) // sqrt_Q + 1)
    # Adjust P to be in proper range
    P = sqrt_kn - ((sqrt_kn - P) % sqrt_Q)
    Q = (kn - P * P) // Q_prev
    
    if Q == 0:
        return gcd(P, n)
    
    # Continue reverse cycle until P_prev == P (symmetry point)
    for _ in range(max_iterations):
        b = (sqrt_kn + P) // Q
        P_new = b * Q - P
        Q_new = Q_prev + b * (P - P_new)
        
        if P == P_new:
            # Found symmetry point - extract factor
            factor = gcd(P, n)
            if 1 < factor < n:
                return factor
            factor = gcd(Q, n)
            if 1 < factor < n:
                return factor
            break
        
        P_prev, Q_prev = P, Q
        P, Q = P_new, Q_new
        
        if Q == 0:
            break
    
    return None


def squfof_factor(n: int, max_iterations: int = 1000000) -> Tuple[int, int]:
    """
    Factor n using SQUFOF.
    
    Returns:
        (p, q) where p * q = n, or (n, 1) if factorization failed
    """
    if n <= 1:
        return (n, 1)
    if n % 2 == 0:
        return (2, n // 2)
    if is_prime(n):
        return (n, 1)
    
    factor = squfof(n, max_iterations)
    if factor is not None:
        return (factor, n // factor)
    
    return (n, 1)


def squfof_racing(n: int) -> Optional[int]:
    """
    Racing SQUFOF: run multiple multipliers in parallel conceptually.
    
    Interleaves iterations across different multipliers to find
    factors faster on average.
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    
    sqrt_n = isqrt(n)
    if sqrt_n * sqrt_n == n:
        return sqrt_n
    
    multipliers = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Initialize state for each multiplier
    states = []
    for k in multipliers:
        kn = k * n
        sqrt_kn = isqrt(kn)
        if sqrt_kn * sqrt_kn == kn:
            continue
        
        P = sqrt_kn
        Q = kn - P * P
        if Q == 0:
            continue
        
        states.append({
            'k': k,
            'kn': kn,
            'sqrt_kn': sqrt_kn,
            'P': P,
            'Q': Q,
            'Q_prev': 1,
            'iteration': 0,
            'phase': 'forward',
            'sqrt_Q': 0
        })
    
    # Race all multipliers
    max_iterations = 100000
    for global_iter in range(max_iterations):
        for state in states:
            if state['phase'] == 'done':
                continue
            
            state['iteration'] += 1
            kn = state['kn']
            sqrt_kn = state['sqrt_kn']
            P, Q = state['P'], state['Q']
            Q_prev = state['Q_prev']
            
            # One iteration
            b = (sqrt_kn + P) // Q
            P_new = b * Q - P
            Q_new = Q_prev + b * (P - P_new)
            
            if state['phase'] == 'forward':
                # Check for square
                if state['iteration'] % 2 == 0:
                    sqrt_Q = isqrt(Q)
                    if sqrt_Q * sqrt_Q == Q:
                        state['sqrt_Q'] = sqrt_Q
                        state['phase'] = 'reverse'
                        # Initialize reverse
                        state['Q_prev'] = sqrt_Q
                        state['P'] = sqrt_kn - ((sqrt_kn - P) % sqrt_Q)
                        state['Q'] = (kn - state['P'] * state['P']) // sqrt_Q
                        continue
            
            elif state['phase'] == 'reverse':
                if P == P_new:
                    # Symmetry - extract factor
                    factor = gcd(P, n)
                    if 1 < factor < n:
                        return factor
                    factor = gcd(Q, n)
                    if 1 < factor < n:
                        return factor
                    state['phase'] = 'done'
                    continue
            
            state['P'], state['Q'] = P_new, Q_new
            state['Q_prev'] = Q
            
            if Q_new == 0:
                state['phase'] = 'done'
    
    return None

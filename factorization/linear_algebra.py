"""
Linear Algebra over GF(2) for Factorization Algorithms.

The sieving phase of QS/MPQS/GNFS produces relations of the form:
    x² ≡ y (mod n) where y = ∏ p_i^{e_i}

We need to find a subset of relations whose product gives:
    X² ≡ Y² (mod n)

This requires finding vectors in the null space of the exponent
matrix over GF(2) (since we only care if exponents are even/odd).

Algorithms implemented:
    - Gaussian elimination (simple, O(n³))
    - Block Lanczos (for large sparse matrices, used in GNFS)
    - Structured Gaussian elimination with large prime handling

References:
    - Pomerance, C. "The Quadratic Sieve Factoring Algorithm"
    - LaMacchia, B. & Odlyzko, A. "Solving Large Sparse Linear Systems over GF(2)"
"""

import numpy as np
from typing import List, Tuple, Optional
import random


def gaussian_elimination_gf2(matrix: np.ndarray) -> List[List[int]]:
    """
    Gaussian elimination over GF(2) to find null space.
    
    Given an m×k matrix M over GF(2), finds all vectors v such that M·v = 0.
    
    Args:
        matrix: numpy array with entries 0 or 1 (m relations × k primes)
    
    Returns:
        List of null space vectors (each vector is a list of 0s and 1s)
    
    Time complexity: O(m * k * min(m, k))
    
    Example:
        >>> M = np.array([[1,0,1], [1,1,0], [0,1,1]])
        >>> null = gaussian_elimination_gf2(M)
        >>> # Should find that row1 + row2 + row3 = 0
    """
    m, k = matrix.shape
    
    # Work on a copy, augmented with identity to track row operations
    aug = np.zeros((m, k + m), dtype=np.int8)
    aug[:, :k] = matrix % 2
    aug[:, k:] = np.eye(m, dtype=np.int8)
    
    # Forward elimination
    pivot_row = 0
    pivot_cols = []
    
    for col in range(k):
        # Find pivot in this column
        pivot_found = False
        for row in range(pivot_row, m):
            if aug[row, col] == 1:
                # Swap to pivot position
                aug[[pivot_row, row]] = aug[[row, pivot_row]]
                pivot_found = True
                break
        
        if not pivot_found:
            continue
        
        pivot_cols.append(col)
        
        # Eliminate this column in all other rows
        for row in range(m):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[pivot_row]) % 2
        
        pivot_row += 1
        if pivot_row >= m:
            break
    
    # Find null space from rows that became zero in the left part
    null_vectors = []
    for row in range(m):
        if np.sum(aug[row, :k]) == 0:
            # This row's right part gives a dependency
            dependency = aug[row, k:].tolist()
            if sum(dependency) > 0:  # Non-trivial dependency
                null_vectors.append(dependency)
    
    return null_vectors


def find_dependencies(relations: List[Tuple], factor_base: List[int]) -> List[List[int]]:
    """
    Find linear dependencies among relation exponent vectors over GF(2).
    
    Args:
        relations: List of (x, y²) pairs where y² is smooth over factor_base
        factor_base: List of primes in the factor base
    
    Returns:
        List of dependency vectors indicating which relations to combine
    
    Each dependency vector has length = len(relations), with 1s indicating
    which relations to multiply together to get a perfect square.
    """
    from .utils import factor_over_base
    
    # Build exponent matrix
    rows = []
    valid_indices = []
    
    for i, (x, y_sq) in enumerate(relations):
        exp_vec = factor_over_base(y_sq, factor_base)
        if exp_vec is not None:
            # Take mod 2 for GF(2) arithmetic
            rows.append([e % 2 for e in exp_vec])
            valid_indices.append(i)
    
    if len(rows) < len(factor_base):
        return []
    
    matrix = np.array(rows, dtype=np.int8)
    null_vectors = gaussian_elimination_gf2(matrix)
    
    # Map back to original relation indices
    result = []
    for null_vec in null_vectors:
        full_vec = [0] * len(relations)
        for i, bit in enumerate(null_vec):
            if bit:
                full_vec[valid_indices[i]] = 1
        result.append(full_vec)
    
    return result


def block_lanczos_gf2(matrix: np.ndarray, block_size: int = 64) -> List[List[int]]:
    """
    Block Lanczos algorithm for finding null space over GF(2).
    
    More efficient than Gaussian elimination for large sparse matrices,
    with complexity O(weight * n) where weight is number of non-zeros.
    
    This is a simplified implementation. Production code would use
    bit-packed arithmetic and cache-optimized block operations.
    
    Args:
        matrix: Sparse binary matrix (m×k)
        block_size: Block width (typically 32 or 64 for word operations)
    
    Returns:
        List of null space vectors
    
    Note: Full implementation would use Montgomery's structured approach
    with careful handling of the Lanczos iteration's termination.
    """
    m, k = matrix.shape
    
    # For small matrices, fall back to Gaussian elimination
    if m < 1000 or k < 1000:
        return gaussian_elimination_gf2(matrix)
    
    # Simplified Block Lanczos
    # In practice, would use 64-bit words for blocks
    
    # Start with random block
    Y = np.random.randint(0, 2, size=(k, block_size), dtype=np.int8)
    
    V_prev = np.zeros((k, block_size), dtype=np.int8)
    V = Y.copy()
    
    null_vectors = []
    max_iterations = k // block_size + 10
    
    for iteration in range(max_iterations):
        # W = A^T * A * V
        AV = (matrix @ V) % 2
        AtAV = (matrix.T @ AV) % 2
        
        # Check for null space elements
        for j in range(block_size):
            col = V[:, j]
            if np.sum((matrix @ col) % 2) == 0:
                vec = col.tolist()
                if sum(vec) > 0 and vec not in null_vectors:
                    null_vectors.append(vec)
        
        if len(null_vectors) >= 10:
            break
        
        # Update V for next iteration (simplified)
        V_new = (AtAV + V_prev) % 2
        V_prev = V.copy()
        V = V_new
        
        # Orthogonalize (simplified)
        for j in range(block_size):
            for i in range(j):
                if np.sum((V[:, i] * V[:, j]) % 2) > 0:
                    V[:, j] = (V[:, j] + V[:, i]) % 2
    
    return null_vectors


def structured_gaussian_gf2(matrix: np.ndarray, 
                            structured_rows: Optional[List[int]] = None) -> List[List[int]]:
    """
    Structured Gaussian elimination for matrices with special structure.
    
    When some relations involve "large primes" that appear in few relations,
    we can process them first using "singleton removal" and structured
    elimination, reducing the dense matrix size.
    
    This is the approach used in practical QS/MPQS implementations.
    
    Args:
        matrix: Binary exponent matrix
        structured_rows: Indices of rows with special structure (e.g., large primes)
    
    Returns:
        List of null space vectors
    """
    # For now, delegate to standard Gaussian elimination
    # Full implementation would:
    # 1. Remove singletons (columns with only one 1)
    # 2. Handle large primes with merge step
    # 3. Only do dense elimination on reduced core matrix
    return gaussian_elimination_gf2(matrix)


def random_combination_search(relations: List[Tuple], factor_base: List[int],
                              n: int, max_attempts: int = 1000) -> Optional[int]:
    """
    Probabilistic search for dependencies using random combinations.
    
    Sometimes faster than full linear algebra when we only need one dependency.
    Birthday paradox: with k relations over a k-dimensional space, random
    combinations have good chance of hitting the null space.
    
    Args:
        relations: List of (x, y²) pairs
        factor_base: Prime factor base
        n: Number being factored
        max_attempts: Maximum random combinations to try
    
    Returns:
        A factor of n if found, None otherwise
    """
    from .utils import factor_over_base, gcd
    
    # Pre-factor all relations
    factored = []
    for x, y_sq in relations:
        exp_vec = factor_over_base(y_sq, factor_base)
        if exp_vec is not None:
            factored.append((x, y_sq, exp_vec))
    
    if len(factored) < 2:
        return None
    
    for _ in range(max_attempts):
        # Random subset size (birthday paradox suggests sqrt(k) is good)
        subset_size = random.randint(2, min(20, len(factored)))
        indices = random.sample(range(len(factored)), subset_size)
        
        # Compute combined exponent vector
        x_prod = 1
        total_exp = [0] * len(factor_base)
        
        for idx in indices:
            x, y_sq, exp_vec = factored[idx]
            x_prod = (x_prod * x) % n
            for j, e in enumerate(exp_vec):
                total_exp[j] += e
        
        # Check if all exponents are even
        if all(e % 2 == 0 for e in total_exp):
            # Compute y = sqrt(∏ y_i²)
            y = 1
            for j, e in enumerate(total_exp):
                y = (y * pow(factor_base[j], e // 2, n)) % n
            
            factor = gcd(x_prod - y, n)
            if 1 < factor < n:
                return factor
            
            factor = gcd(x_prod + y, n)
            if 1 < factor < n:
                return factor
    
    return None

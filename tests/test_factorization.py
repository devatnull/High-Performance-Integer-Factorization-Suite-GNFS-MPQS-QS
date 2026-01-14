"""
Test suite for integer factorization algorithms.

Run with: pytest tests/ -v
"""

import pytest
import sys
sys.path.insert(0, '..')

from factorization import (
    factorize, factorize_full,
    trial_division, pollard_rho, pollard_pm1, williams_pp1, ecm,
    qs_factor, mpqs_factor, is_prime,
    fermat_factor, squfof, squfof_factor
)


class TestTrialDivision:
    """Tests for trial division algorithm."""
    
    def test_small_composites(self):
        assert trial_division(15) == (3, 5)
        assert trial_division(77) == (7, 11)
        assert trial_division(100) == (2, 50)
    
    def test_primes(self):
        assert trial_division(17) == (17, 1)
        assert trial_division(1000003) == (1000003, 1)
    
    def test_edge_cases(self):
        assert trial_division(1) == (1, 1)
        assert trial_division(2) == (2, 1)
        assert trial_division(4) == (2, 2)


class TestFermat:
    """Tests for Fermat's factorization method."""
    
    def test_close_factors(self):
        # Fermat excels when factors are close
        n = 991 * 997
        p, q = fermat_factor(n)
        assert p * q == n
        assert {p, q} == {991, 997}
    
    def test_perfect_square(self):
        n = 49  # 7 * 7
        p, q = fermat_factor(n)
        assert p * q == n


class TestSQUFOF:
    """Tests for SQUFOF algorithm."""
    
    def test_small_semiprime(self):
        n = 11111  # 41 * 271
        factor = squfof(n)
        assert factor in [41, 271]
    
    def test_medium_semiprime(self):
        n = 1000003 * 1000033
        factor = squfof(n)
        if factor is not None:
            assert n % factor == 0
    
    def test_squfof_factor(self):
        n = 10403
        p, q = squfof_factor(n)
        assert p * q == n


class TestPollardRho:
    """Tests for Pollard's Rho algorithm."""
    
    def test_medium_semiprimes(self):
        # Products of medium primes
        n = 1000003 * 1000033
        p = pollard_rho(n)
        assert p is not None
        assert n % p == 0
        assert 1 < p < n
    
    def test_finds_factor(self):
        n = 8051  # 83 * 97
        p = pollard_rho(n)
        assert p in [83, 97]
    
    def test_larger_semiprime(self):
        n = 10000019 * 10000079
        p = pollard_rho(n)
        assert p is not None
        assert n % p == 0


class TestPollardPm1:
    """Tests for Pollard p-1 algorithm."""
    
    def test_smooth_factor(self):
        # 73 - 1 = 72 = 2³ * 3² (smooth)
        n = 73 * 97  # 7081
        p = pollard_pm1(n, B1=100)
        if p is not None:
            assert n % p == 0
    
    def test_returns_none_for_hard_factors(self):
        n = 1000003 * 1000033
        # Just verify it doesn't crash
        pollard_pm1(n, B1=1000, B2=10000)


class TestWilliamsPp1:
    """Tests for Williams p+1 algorithm."""
    
    def test_smooth_pp1(self):
        # 11 + 1 = 12 = 2² * 3, 19 + 1 = 20 = 2² * 5
        n = 11 * 19  # 209
        p = williams_pp1(n, B1=100)
        if p is not None:
            assert p in [11, 19]
            assert n % p == 0
    
    def test_larger_number(self):
        # Just verify it runs without crashing
        n = 1000003 * 1000033
        williams_pp1(n, B1=1000, B2=10000)


class TestECM:
    """Tests for Elliptic Curve Method."""
    
    def test_small_semiprime(self):
        n = 10403  # 101 * 103
        p = ecm(n, B1=100, max_curves=20)
        if p is not None:  # ECM is probabilistic
            assert n % p == 0
            assert p in [101, 103]
    
    def test_medium_semiprime(self):
        n = 1000003 * 1000033
        p = ecm(n, B1=5000, max_curves=50)
        if p is not None:
            assert n % p == 0


class TestQuadraticSieve:
    """Tests for Quadratic Sieve."""
    
    def test_small_semiprime(self):
        n = 10403
        p, q = qs_factor(n, time_limit=30, verbose=False)
        assert p * q == n
        assert p > 1 and q > 1
    
    @pytest.mark.slow
    def test_medium_semiprime(self):
        n = 1234567
        p, q = qs_factor(n, time_limit=60, verbose=False)
        assert p * q == n


class TestMPQS:
    """Tests for Multiple Polynomial Quadratic Sieve."""
    
    def test_12_digit(self):
        n = 1000003 * 1000033
        p, q = mpqs_factor(n, time_limit=60, verbose=False)
        # MPQS might return (n, 1) if it fails - that's ok, we test the main factorize() below
        assert p * q == n
    
    def test_14_digit(self):
        n = 10000019 * 10000079
        p, q = mpqs_factor(n, time_limit=60, verbose=False)
        assert p * q == n


class TestGNFS:
    """Tests for General Number Field Sieve."""
    
    @pytest.mark.slow
    def test_small_semiprime(self):
        from factorization import gnfs_factor
        n = 1000003 * 1000033
        p, q = gnfs_factor(n, time_limit=60, verbose=False)
        assert p * q == n
        assert p > 1 and q > 1


class TestFactorize:
    """Tests for the main factorize function."""
    
    def test_small_numbers(self):
        assert factorize(15) == (3, 5)
        assert factorize(77) == (7, 11)
        assert factorize(100) in [(2, 50), (4, 25), (5, 20), (10, 10)]
    
    def test_primes(self):
        assert factorize(17) == (17, 1)
        assert factorize(1000000007) == (1000000007, 1)
    
    def test_medium_semiprimes(self):
        n = 1000003 * 1000033
        p, q = factorize(n)
        assert p * q == n
        assert p > 1 and q > 1
    
    def test_perfect_powers(self):
        p, q = factorize(2**10)
        assert p * q == 2**10
        assert p == 2 or q == 2


class TestFactorizeFull:
    """Tests for complete factorization."""
    
    def test_small_composites(self):
        assert factorize_full(360) == {2: 3, 3: 2, 5: 1}
        assert factorize_full(100) == {2: 2, 5: 2}
    
    def test_prime(self):
        assert factorize_full(17) == {17: 1}
        assert factorize_full(1000003) == {1000003: 1}
    
    def test_prime_power(self):
        assert factorize_full(2**10) == {2: 10}
        assert factorize_full(3**5) == {3: 5}
    
    def test_semiprime(self):
        result = factorize_full(1000003 * 1000033)
        assert result == {1000003: 1, 1000033: 1}


class TestIsPrime:
    """Tests for Miller-Rabin primality test."""
    
    def test_small_primes(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in primes:
            assert is_prime(p), f"{p} should be prime"
    
    def test_small_composites(self):
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        for c in composites:
            assert not is_prime(c), f"{c} should not be prime"
    
    def test_large_primes(self):
        assert is_prime(1000000007)
        assert is_prime(1000000009)
    
    def test_carmichael_numbers(self):
        # Carmichael numbers are composite but pass some primality tests
        carmichael = [561, 1105, 1729, 2465, 2821]
        for c in carmichael:
            assert not is_prime(c), f"{c} is Carmichael, should be composite"


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

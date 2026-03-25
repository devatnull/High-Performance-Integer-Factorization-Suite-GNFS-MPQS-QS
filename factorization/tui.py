#!/usr/bin/env python3
"""
Interactive TUI for Integer Factorization Suite.

Run with: python -m factorization.tui
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
import textwrap
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import (
    ecm,
    estimate_gnfs_runtime,
    factorize,
    factorize_full,
    gmpy2_available,
    gnfs_factor,
    is_prime,
    mpqs_factor,
    numba_available,
    pollard_pm1,
    pollard_rho,
    qs_factor,
    trial_division,
    williams_pp1,
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


ColorName = str
FactorPair = Tuple[int, int]
RunResult = Tuple[str, object]


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    slug: str
    name: str
    best_for: str
    range_hint: str
    strengths: str
    caveats: str
    result_shape: str
    experimental: bool = False


@dataclass(frozen=True)
class NumberAnalysis:
    n: int
    digits: int
    bits: int
    probable_prime: bool
    divisibility_hints: List[str]
    recommendation_keys: List[str]
    recommendation_reason: str


ALGORITHM_SPECS: Dict[str, AlgorithmSpec] = {
    "1": AlgorithmSpec(
        key="1",
        slug="trial",
        name="Trial Division",
        best_for="Very small inputs and obvious small factors.",
        range_hint="Best below roughly 10^12.",
        strengths="Deterministic, simple, and excellent at clearing tiny factors fast.",
        caveats="Becomes unusably slow as digit count grows.",
        result_shape="Can return a composite cofactor if the first factor found is small.",
    ),
    "2": AlgorithmSpec(
        key="2",
        slug="fermat",
        name="Fermat",
        best_for="Numbers whose factors are close together.",
        range_hint="Useful when p ≈ q.",
        strengths="Very fast on balanced semiprimes with small factor gap.",
        caveats="Poor fit when factors are far apart or one factor is small.",
        result_shape="Returns one split; not a complete factorization method.",
    ),
    "3": AlgorithmSpec(
        key="3",
        slug="rho",
        name="Pollard Rho",
        best_for="General-purpose medium-size composites.",
        range_hint="Usually strongest in the small-to-mid digit range.",
        strengths="Excellent default when you suspect a modest-size factor exists.",
        caveats="Probabilistic; may need more iterations or a different method.",
        result_shape="Finds one non-trivial factor, then the TUI shows the paired cofactor.",
    ),
    "4": AlgorithmSpec(
        key="4",
        slug="squfof",
        name="SQUFOF",
        best_for="Small and medium semiprimes with very low overhead.",
        range_hint="Competitive with Pollard Rho on modest inputs.",
        strengths="Constant-memory, quick startup, and good for quick checks.",
        caveats="Not the right tool once inputs move into QS/MPQS territory.",
        result_shape="Returns one split; not recursive by itself.",
    ),
    "5": AlgorithmSpec(
        key="5",
        slug="pm1",
        name="Pollard p-1",
        best_for="Factors where p-1 is smooth.",
        range_hint="Special-purpose method; input size matters less than smoothness.",
        strengths="Can beat generic methods by a lot when the structure matches.",
        caveats="Often fails completely when p-1 is not smooth enough.",
        result_shape="Finds one factor if the smoothness assumption holds.",
    ),
    "6": AlgorithmSpec(
        key="6",
        slug="pp1",
        name="Williams p+1",
        best_for="Factors where p+1 is smooth.",
        range_hint="Special-purpose, similar role to p-1 with different structure.",
        strengths="Useful when p-1 misses but p+1 happens to be smooth.",
        caveats="Still highly structure-dependent and not a general default.",
        result_shape="Finds one factor when the Lucas-sequence path succeeds.",
    ),
    "7": AlgorithmSpec(
        key="7",
        slug="ecm",
        name="ECM",
        best_for="Extracting medium-size factors from large composites.",
        range_hint="Often the best bridge between Pollard methods and QS/MPQS.",
        strengths="Runtime depends mostly on factor size, not the full composite size.",
        caveats="Probabilistic and parameter-sensitive; may need larger bounds or more curves.",
        result_shape="Finds one factor; excellent before heavier sieve methods.",
    ),
    "8": AlgorithmSpec(
        key="8",
        slug="qs",
        name="Quadratic Sieve",
        best_for="Moderate semiprimes when Pollard/ECM are no longer ideal.",
        range_hint="Roughly the lower edge of sieve-based factoring.",
        strengths="Solid general sieve for medium inputs with predictable behavior.",
        caveats="Outgrown by MPQS on larger targets.",
        result_shape="Aims to produce a full split of a semiprime.",
    ),
    "9": AlgorithmSpec(
        key="9",
        slug="mpqs",
        name="MPQS",
        best_for="Hard semiprimes in the practical single-machine sieve range.",
        range_hint="Roughly 25-80 digits, depending on structure and hardware.",
        strengths="Best heavy-duty method in this project before GNFS, with adaptive multicore sieving.",
        caveats="Still sensitive to input shape and optional backends; not magic on every composite.",
        result_shape="Aims to split the composite directly after relation collection and linear algebra.",
    ),
    "0": AlgorithmSpec(
        key="0",
        slug="gnfs",
        name="GNFS",
        best_for="Very large semiprimes where asymptotics matter most.",
        range_hint="Experimental path for the largest inputs in this repo.",
        strengths="The asymptotically best classical algorithm in the suite.",
        caveats="Experimental, expensive, and not a casual interactive default.",
        result_shape="Attempts a full split, but failures or fallback behavior are possible.",
        experimental=True,
    ),
}


def _wrap_single_factor(func: Callable[..., Optional[int]], n: int, **kwargs: object) -> FactorPair:
    """Wrap factor-returning algorithms into a pair."""
    factor = func(n, **kwargs)
    if factor and factor not in (1, n) and n % factor == 0:
        return (factor, n // factor)
    return (n, 1)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_colored(text: str, color: ColorName = "white") -> None:
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def format_factorization(factors: Dict[int, int]) -> str:
    """Format factorization as a human-readable product."""
    if not factors:
        return "1"

    parts = []
    for prime in sorted(factors):
        exponent = factors[prime]
        parts.append(f"{prime}^{exponent}" if exponent > 1 else str(prime))
    return " × ".join(parts)


def render_box(title: str, lines: Sequence[str], color: ColorName = "cyan") -> None:
    width = 77
    print_colored("┌" + "─" * width + "┐", color)
    print_colored(f"│ {title.upper():<{width - 1}}│", color)
    print_colored("├" + "─" * width + "┤", color)
    for line in lines:
        wrapped = textwrap.wrap(
            line,
            width=width - 1,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        for text in wrapped:
            print(f"│ {text:<{width - 1}}│")
    print_colored("└" + "─" * width + "┘", color)


def pause() -> None:
    print()
    input("Press Enter to continue...")


def parse_number_text(text: str) -> int:
    """Parse a safe positive integer input.

    Accepted formats:
        123456789
        10^20
        10**20
        1e12
    """
    raw = text.strip().replace("_", "")
    if not raw:
        raise ValueError("empty input")

    if re.fullmatch(r"\d+", raw):
        value = int(raw)
    else:
        power_match = re.fullmatch(r"(\d+)\s*(\^|\*\*)\s*(\d+)", raw)
        sci_match = re.fullmatch(r"(\d+)[eE]([+-]?\d+)", raw)

        if power_match:
            base = int(power_match.group(1))
            exponent = int(power_match.group(3))
            value = pow(base, exponent)
        elif sci_match:
            mantissa = int(sci_match.group(1))
            exponent = int(sci_match.group(2))
            if exponent < 0:
                raise ValueError("scientific notation must evaluate to an integer >= 1")
            value = mantissa * pow(10, exponent)
        else:
            raise ValueError("unsupported format")

    if value < 1:
        raise ValueError("number must be positive")
    return value


def get_number(prompt: str = "Enter number: ") -> Optional[int]:
    try:
        text = input(prompt).strip()
    except KeyboardInterrupt:
        return None

    if not text:
        return None

    try:
        return parse_number_text(text)
    except ValueError:
        print_colored(
            "Invalid number format. Use decimal integers, 10^20, 10**20, or 1e12.",
            "red",
        )
        return None


def prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            text = input(f"{prompt} [{suffix}]: ").strip().lower()
        except KeyboardInterrupt:
            return default
        if not text:
            return default
        if text in {"y", "yes"}:
            return True
        if text in {"n", "no"}:
            return False
        print_colored("Please answer yes or no.", "yellow")


def prompt_int(prompt: str, default: int, minimum: int = 1, allow_blank: bool = True) -> int:
    while True:
        try:
            text = input(f"{prompt} [{default}]: ").strip()
        except KeyboardInterrupt:
            return default
        if not text and allow_blank:
            return default
        try:
            value = int(text)
        except ValueError:
            print_colored("Please enter an integer.", "yellow")
            continue
        if value < minimum:
            print_colored(f"Value must be >= {minimum}.", "yellow")
            continue
        return value


def prompt_optional_int(prompt: str, default: Optional[int], minimum: int = 1) -> Optional[int]:
    default_text = "auto" if default is None else str(default)
    while True:
        try:
            text = input(f"{prompt} [{default_text}]: ").strip().lower()
        except KeyboardInterrupt:
            return default
        if not text:
            return default
        if text == "auto":
            return None
        try:
            value = int(text)
        except ValueError:
            print_colored("Enter an integer or 'auto'.", "yellow")
            continue
        if value < minimum:
            print_colored(f"Value must be >= {minimum}.", "yellow")
            continue
        return value


def prompt_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        try:
            text = input(f"{prompt} [{default}]: ").strip()
        except KeyboardInterrupt:
            return default
        if not text:
            return default
        try:
            value = float(text)
        except ValueError:
            print_colored("Please enter a number.", "yellow")
            continue
        if value < minimum:
            print_colored(f"Value must be >= {minimum}.", "yellow")
            continue
        return value


def parse_time_limit_text(text: str) -> float:
    raw = text.strip().lower()
    if raw in {"none", "inf", "infinite", "unlimited", "no-limit", "nolimit"}:
        return math.inf

    value = float(raw)
    if value < 1.0:
        raise ValueError("time limit must be >= 1 second")
    return value


def format_time_limit(value: float) -> str:
    if math.isinf(value):
        return "none"
    return str(value)


def prompt_time_limit(prompt: str, default: float) -> float:
    while True:
        try:
            text = input(f"{prompt} [{format_time_limit(default)}]: ").strip()
        except KeyboardInterrupt:
            return default
        if not text:
            return default
        try:
            return parse_time_limit_text(text)
        except ValueError:
            print_colored("Enter seconds or 'none' for no limit.", "yellow")


def get_algorithm_specs() -> List[AlgorithmSpec]:
    return [ALGORITHM_SPECS[key] for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]]


def classify_probable_algorithm_band(digits: int) -> Tuple[List[str], str]:
    if digits < 10:
        return (
            ["1", "3", "4"],
            "Small input: inexpensive direct methods should dominate.",
        )
    if digits < 20:
        return (
            ["3", "7", "4"],
            "Medium input: Pollard-style methods and ECM are usually the right first pass.",
        )
    if digits < 30:
        return (
            ["7", "9", "8"],
            "This is crossing into sieve territory; ECM is still useful for medium-size factors.",
        )
    if digits < 60:
        return (
            ["9", "7", "8"],
            "This is prime MPQS territory for a single machine, with ECM still useful as a factor extractor.",
        )
    return (
        ["0", "9", "7"],
        "The number is large enough that GNFS is the asymptotic play, but MPQS and ECM can still be useful depending on factor structure.",
    )


def analyze_number(n: int) -> NumberAnalysis:
    digits = len(str(n))
    hints: List[str] = []
    small_divisors = [2, 3, 5, 7, 11, 13]
    for divisor in small_divisors:
        if n > divisor and n % divisor == 0:
            hints.append(f"divisible by {divisor}")

    if not hints:
        if n % 2:
            hints.append("odd")
        if sum(int(ch) for ch in str(n)) % 3 != 0:
            hints.append("not divisible by 3")
        if not str(n).endswith(("0", "5")):
            hints.append("not divisible by 5")

    probable_prime = is_prime(n)
    recommendation_keys, reason = classify_probable_algorithm_band(digits)
    if probable_prime:
        reason = "This input is already prime under the current primality test, so factorization algorithms will only confirm that."
        recommendation_keys = ["3", "7"]

    return NumberAnalysis(
        n=n,
        digits=digits,
        bits=n.bit_length(),
        probable_prime=probable_prime,
        divisibility_hints=hints,
        recommendation_keys=recommendation_keys,
        recommendation_reason=reason,
    )


def collect_system_info() -> Dict[str, object]:
    cpu_count = os.cpu_count() or 1
    algorithms = [
        f"{spec.name}{' (experimental)' if spec.experimental else ''}"
        for spec in get_algorithm_specs()
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "numba": numba_available(),
        "gmpy2": gmpy2_available(),
        "mpqs_workers": f"adaptive (1 worker on small jobs, up to {cpu_count} on large MPQS jobs)",
        "algorithms": algorithms,
    }


def benchmark_mode_args(mode: str) -> List[str]:
    mapping = {
        "1": ["--quick"],
        "2": [],
        "3": ["--full"],
    }
    if mode not in mapping:
        raise ValueError("unknown benchmark mode")
    return mapping[mode]


def render_header() -> None:
    clear_screen()
    print_colored(BANNER, "cyan")


def render_main_menu() -> None:
    print(MENU)


def render_number_analysis(analysis: NumberAnalysis) -> None:
    recommended = ", ".join(ALGORITHM_SPECS[key].name for key in analysis.recommendation_keys)
    lines = [
        f"Number: {analysis.n}",
        f"Digits / bits: {analysis.digits} digits, {analysis.bits} bits",
        f"Primality signal: {'probable prime' if analysis.probable_prime else 'composite or likely composite'}",
        f"Quick hints: {', '.join(analysis.divisibility_hints)}",
        f"Recommended next methods: {recommended}",
        f"Why: {analysis.recommendation_reason}",
    ]
    render_box("Number Analysis", lines, color="cyan")


def show_algorithm_catalog() -> None:
    lines: List[str] = []
    for spec in get_algorithm_specs():
        label = f"[{spec.key}] {spec.name}"
        if spec.experimental:
            label += " [experimental]"
        lines.extend(
            [
                label,
                f"    Best for: {spec.best_for}",
                f"    Range:    {spec.range_hint}",
                f"    Strength: {spec.strengths}",
                f"    Caveat:   {spec.caveats}",
                f"    Output:   {spec.result_shape}",
                "",
            ]
        )
    render_box("Algorithm Catalog", lines[:-1], color="cyan")


def choose_algorithm() -> Optional[AlgorithmSpec]:
    show_algorithm_catalog()
    try:
        choice = input("\nSelect algorithm (1-0): ").strip()
    except KeyboardInterrupt:
        return None
    if choice not in ALGORITHM_SPECS:
        print_colored("Invalid choice.", "red")
        return None
    return ALGORITHM_SPECS[choice]


def get_algorithm_defaults(spec: AlgorithmSpec) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "verbose": True,
        "time_limit": 300.0,
    }
    if spec.slug == "auto":
        defaults["full_factorization"] = True
    if spec.slug == "rho":
        defaults["max_iterations"] = 1_000_000
    elif spec.slug == "ecm":
        defaults.update({"B1": 50000, "B2": 5_000_000, "max_curves": 100})
    elif spec.slug == "qs":
        defaults["time_limit"] = 300.0
    elif spec.slug == "mpqs":
        defaults.update({"time_limit": 300.0, "num_workers": None})
    elif spec.slug == "gnfs":
        defaults["time_limit"] = 600.0
    return defaults


def collect_auto_config() -> Dict[str, object]:
    return {
        "verbose": prompt_yes_no("Verbose output", True),
        "full_factorization": prompt_yes_no("Show complete prime-power factorization", True),
        "time_limit": prompt_time_limit("Time limit in seconds", 300.0),
    }


def collect_algorithm_config(spec: AlgorithmSpec) -> Optional[Dict[str, object]]:
    if spec.slug == "auto":
        return collect_auto_config()

    config: Dict[str, object] = {"verbose": True, "time_limit": 300.0}

    if spec.slug == "trial":
        return {}
    if spec.slug == "fermat":
        return {}
    if spec.slug == "rho":
        return {
            "max_iterations": prompt_int("Max iterations", 1_000_000, minimum=1),
        }
    if spec.slug == "squfof":
        return {}
    if spec.slug == "pm1":
        return {
            "B1": prompt_int("Stage 1 bound B1", 10000, minimum=2),
            "B2": prompt_int("Stage 2 bound B2", 1_000_000, minimum=2),
        }
    if spec.slug == "pp1":
        return {
            "B1": prompt_int("Stage 1 bound B1", 10000, minimum=2),
            "B2": prompt_int("Stage 2 bound B2", 1_000_000, minimum=2),
        }
    if spec.slug == "ecm":
        return {
            "B1": prompt_int("Stage 1 bound B1", 50000, minimum=2),
            "B2": prompt_int("Stage 2 bound B2", 5_000_000, minimum=2),
            "max_curves": prompt_int("Maximum curves", 100, minimum=1),
        }
    if spec.slug == "qs":
        return {
            "time_limit": prompt_time_limit("Time limit in seconds", 300.0),
            "verbose": prompt_yes_no("Verbose output", True),
        }
    if spec.slug == "mpqs":
        return {
            "time_limit": prompt_time_limit("Time limit in seconds", 300.0),
            "verbose": prompt_yes_no("Verbose output", True),
            "num_workers": prompt_optional_int("Worker processes", None, minimum=1),
        }
    if spec.slug == "gnfs":
        print_colored(
            "\nWarning: GNFS is experimental and expensive. It is not a casual interactive default.",
            "yellow",
        )
        proceed = prompt_yes_no("Proceed with GNFS anyway", False)
        if not proceed:
            return None
        return {
            "time_limit": prompt_time_limit("Time limit in seconds", 600.0),
            "verbose": prompt_yes_no("Verbose output", True),
        }
    return config


def execute_auto_factorization(n: int, config: Dict[str, object]) -> RunResult:
    verbose = bool(config["verbose"])
    time_limit = float(config["time_limit"])
    if config["full_factorization"]:
        factors = factorize_full(n, time_limit=time_limit, verbose=verbose)
        return ("full", factors)
    return ("pair", factorize(n, time_limit=time_limit, verbose=verbose))


def execute_specific_algorithm(spec: AlgorithmSpec, n: int, config: Dict[str, object]) -> RunResult:
    slug = spec.slug
    if slug == "trial":
        return ("pair", trial_division(n))
    if slug == "fermat":
        return ("pair", fermat_factor(n))
    if slug == "rho":
        return ("pair", _wrap_single_factor(pollard_rho, n, max_iterations=int(config["max_iterations"])))
    if slug == "squfof":
        return ("pair", squfof_factor(n))
    if slug == "pm1":
        return ("pair", _wrap_single_factor(pollard_pm1, n, B1=int(config["B1"]), B2=int(config["B2"])))
    if slug == "pp1":
        return ("pair", _wrap_single_factor(williams_pp1, n, B1=int(config["B1"]), B2=int(config["B2"])))
    if slug == "ecm":
        return (
            "pair",
            _wrap_single_factor(
                ecm,
                n,
                B1=int(config["B1"]),
                B2=int(config["B2"]),
                max_curves=int(config["max_curves"]),
            ),
        )
    if slug == "qs":
        return (
            "pair",
            qs_factor(
                n,
                time_limit=float(config["time_limit"]),
                verbose=bool(config["verbose"]),
            ),
        )
    if slug == "mpqs":
        return (
            "pair",
            mpqs_factor(
                n,
                time_limit=float(config["time_limit"]),
                verbose=bool(config["verbose"]),
                num_workers=config["num_workers"],
            ),
        )
    if slug == "gnfs":
        return (
            "pair",
            gnfs_factor(
                n,
                time_limit=float(config["time_limit"]),
                verbose=bool(config["verbose"]),
            ),
        )
    raise ValueError(f"Unsupported algorithm: {slug}")


def format_next_steps(analysis: NumberAnalysis, chosen_name: str) -> str:
    suggestions = [
        ALGORITHM_SPECS[key].name
        for key in analysis.recommendation_keys
        if ALGORITHM_SPECS[key].name != chosen_name
    ]
    if suggestions:
        return f"Try {', '.join(suggestions[:2])} next."
    if analysis.probable_prime:
        return "The input looks prime; use the primality test screen to confirm."
    return "Try a longer timeout or a heavier algorithm."


def show_result_screen(
    n: int,
    algorithm_name: str,
    analysis: NumberAnalysis,
    elapsed: float,
    result_type: str,
    payload: object,
) -> None:
    lines = [
        f"Algorithm: {algorithm_name}",
        f"Elapsed:   {elapsed:.3f}s",
        f"Input:     {n}",
    ]

    if result_type == "full":
        factors = payload if isinstance(payload, dict) else {}
        if factors:
            lines.append(f"Result:    {n} = {format_factorization(factors)}")
            lines.append("Status:    complete prime-power factorization")
        else:
            lines.append("Status:    no factors produced")
    else:
        p, q = payload if isinstance(payload, tuple) else (n, 1)
        if p not in (0, 1) and q not in (0, 1) and p * q == n and p != n:
            p, q = min(p, q), max(p, q)
            lines.append(f"Result:    {n} = {p} × {q}")
            if not (is_prime(p) and is_prime(q)):
                lines.append(f"Prime powers: {format_factorization(factorize_full(n))}")
            lines.append("Status:    non-trivial factors recovered")
        elif analysis.probable_prime:
            lines.append(f"Status:    {n} is prime under the current primality test")
        else:
            lines.append("Status:    factorization attempt did not recover a non-trivial split")
            lines.append(f"Next step: {format_next_steps(analysis, algorithm_name)}")

    render_box("Result", lines, color="green")


def run_auto_flow() -> None:
    print_colored("\n═══ AUTO FACTORIZATION ═══", "cyan")
    n = get_number()
    if n is None:
        return

    analysis = analyze_number(n)
    print()
    render_number_analysis(analysis)
    print()
    config = collect_auto_config()

    start = time.time()
    try:
        result_type, payload = execute_auto_factorization(n, config)
    except KeyboardInterrupt:
        print_colored("\nCancelled", "yellow")
        return
    except Exception as exc:
        print_colored(f"\nError: {exc}", "red")
        return
    elapsed = time.time() - start

    print()
    show_result_screen(n, "Auto", analysis, elapsed, result_type, payload)


def run_specific_algorithm_flow() -> None:
    print_colored("\n═══ SPECIFIC ALGORITHM ═══", "cyan")
    spec = choose_algorithm()
    if spec is None:
        return

    print_colored(f"\nSelected: {spec.name}", "yellow")
    print(f"Best for: {spec.best_for}")
    print(f"Range:    {spec.range_hint}")
    print(f"Caveat:   {spec.caveats}")

    n = get_number()
    if n is None:
        return

    analysis = analyze_number(n)
    print()
    render_number_analysis(analysis)
    print()
    config = collect_algorithm_config(spec)
    if config is None:
        return

    start = time.time()
    try:
        result_type, payload = execute_specific_algorithm(spec, n, config)
    except KeyboardInterrupt:
        print_colored("\nCancelled", "yellow")
        return
    except Exception as exc:
        print_colored(f"\nError: {exc}", "red")
        return
    elapsed = time.time() - start

    print()
    show_result_screen(n, spec.name, analysis, elapsed, result_type, payload)


def run_benchmark_screen() -> None:
    print_colored("\n═══ BENCHMARK ═══", "cyan")
    render_box(
        "Benchmark Modes",
        [
            "[1] Quick    - short MPQS-inclusive benchmark",
            "[2] Standard - default benchmark suite",
            "[3] Extended - fuller benchmark run",
            "",
            f"Numba: {'available' if numba_available() else 'not available'}",
            f"gmpy2: {'available' if gmpy2_available() else 'not available'}",
            f"MPQS worker policy: {collect_system_info()['mpqs_workers']}",
        ],
        color="cyan",
    )

    try:
        choice = input("Select benchmark level (1-3): ").strip()
    except KeyboardInterrupt:
        return

    try:
        args = benchmark_mode_args(choice)
    except ValueError:
        print_colored("Invalid benchmark level.", "red")
        return

    benchmark_script = Path(__file__).resolve().parent.parent / "benchmark.py"
    cmd = [sys.executable, str(benchmark_script), *args]
    print_colored("\nRunning benchmark.py with the current environment...\n", "yellow")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(benchmark_script.parent),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        print_colored(f"Unable to launch benchmark: {exc}", "red")
        return

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        print_colored("\nBenchmark failed:", "red")
        if completed.stderr:
            print(completed.stderr.rstrip())
    elif completed.stderr:
        print_colored("\nBenchmark stderr:", "yellow")
        print(completed.stderr.rstrip())


def estimate_resources() -> None:
    print_colored("\n═══ GNFS RESOURCE ESTIMATOR ═══", "cyan")
    print("Enter a bit size like 512 or a number like 10^100.")
    try:
        text = input("> ").strip()
    except KeyboardInterrupt:
        return

    if not text:
        return

    try:
        if text.isdigit() and len(text) <= 4:
            bits = int(text)
            n = 2 ** bits
            headline = f"Estimating for a {bits}-bit input"
        else:
            n = parse_number_text(text)
            headline = f"Estimating for a {len(str(n))}-digit ({n.bit_length()}-bit) input"
    except ValueError:
        print_colored("Use a bit size or a safe integer expression like 10^100.", "red")
        return

    est = estimate_gnfs_runtime(n)
    lines = [
        headline,
        f"Digits:              {est['digits']}",
        f"Bits:                {est['bits']}",
        f"Estimated core-years:{est['core_years']:.2e}",
        f"Estimated core-hours:{est['core_hours']:.2e}",
        f"Estimated relations: {est['estimated_relations']:.2e}",
        f"Estimated memory GB: {est['estimated_memory_gb']:.2e}",
    ]

    if est["core_years"] < 0.01:
        lines.append("Interpretation: easy on current hardware.")
    elif est["core_years"] < 1:
        lines.append("Interpretation: feasible in hours or days on strong local hardware.")
    elif est["core_years"] < 1000:
        lines.append("Interpretation: cluster-level job, not a casual run.")
    elif est["core_years"] < 1e6:
        lines.append("Interpretation: major research-scale effort.")
    else:
        lines.append("Interpretation: computationally infeasible with known classical methods.")

    render_box("GNFS Estimate", lines, color="cyan")


def check_prime() -> None:
    print_colored("\n═══ PRIMALITY TEST ═══", "cyan")
    n = get_number()
    if n is None:
        return

    analysis = analyze_number(n)
    start = time.time()
    result = is_prime(n)
    elapsed = time.time() - start
    print()
    render_number_analysis(analysis)
    print()
    render_box(
        "Primality Result",
        [
            f"Input:   {n}",
            f"Status:  {'prime' if result else 'composite'}",
            f"Elapsed: {elapsed:.6f}s",
        ],
        color="green" if result else "yellow",
    )


def show_info() -> None:
    print_colored("\n═══ SYSTEM INFO ═══", "cyan")
    info = collect_system_info()
    lines = [
        f"Python:          {info['python']}",
        f"Platform:        {info['platform']}",
        f"CPU cores:       {info['cpu_count']}",
        f"Numba JIT:       {'available' if info['numba'] else 'not available'}",
        f"gmpy2:           {'available' if info['gmpy2'] else 'not available'}",
        f"MPQS workers:    {info['mpqs_workers']}",
        "",
        "Algorithms:",
    ]
    lines.extend([f"  - {name}" for name in info["algorithms"]])
    render_box("System Info", lines, color="cyan")


def show_help() -> None:
    print_colored("\n═══ HELP / OPERATOR GUIDE ═══", "cyan")
    render_box(
        "How To Use The Suite",
        [
            "[f] Auto factor: enter a number, review the analysis, then run with auto-selection.",
            "[a] Specific algorithm: browse the catalog, pick a method, then tune method-specific knobs.",
            "[b] Benchmark: runs the real benchmark.py harness and prints MPQS/backend comparisons.",
            "[e] GNFS estimate: predicts rough compute and memory costs for very large inputs.",
            "[p] Primality test: quick prime/composite verdict with the same safe parser as factoring.",
            "[i] System info: shows Python, CPU, available accelerators, and MPQS worker policy.",
            "",
            "Input formats: decimal integers, 10^20, 10**20, 1e12.",
            "Timeouts: type a number of seconds or 'none' for no time limit.",
            "Auto-selection: small inputs favor direct methods; larger semiprimes push toward ECM/MPQS/GNFS.",
            "Accelerators: gmpy2 speeds big-integer arithmetic; Numba speeds selected hot loops when installed.",
            "TUI vs CLI/API: use the TUI for guided exploration, the CLI for scripts, and the Python API for integration.",
            "GNFS note: the implementation is experimental and should be treated as a research path, not a default workflow.",
        ],
        color="cyan",
    )


def main() -> None:
    render_header()
    while True:
        render_main_menu()
        try:
            choice = input("Command: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if choice == "q":
            print_colored("\nGoodbye!", "cyan")
            break
        if choice == "f":
            run_auto_flow()
        elif choice == "a":
            run_specific_algorithm_flow()
        elif choice == "b":
            run_benchmark_screen()
        elif choice == "e":
            estimate_resources()
        elif choice == "p":
            check_prime()
        elif choice == "i":
            show_info()
        elif choice == "h":
            show_help()
        elif choice == "":
            render_header()
            continue
        else:
            print_colored("Unknown command. Press 'h' for help.", "yellow")

        pause()
        render_header()


if __name__ == "__main__":
    main()

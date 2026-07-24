#!/usr/bin/env python3
"""
surjective_assignments.py
=========================
Counts unique surjective assignments of M resources (of mixed distinguishable
and indistinguishable types) onto n distinguishable targets.

General formula (Principle of Inclusion-Exclusion):

    Omega = sum_{j=0}^{n} (-1)^j * C(n,j) * (n-j)^{M_D}
                           * prod_{i in I} C((n-j) + m_i - 1, m_i)

where:
    m_i  = number of resources of type i
    M_D  = sum of m_i over distinguishable types  (set D)
    I    = set of indistinguishable type indices
    C(a,b) = binomial coefficient

Special cases:
    M < n  →  0          (impossible: too few resources)
    M = n  →  n! / prod_{i in I} m_i!   (bijection closed form)
    M > n  →  full PIE sum

References:
    Feller (1968), Ehrenfest & Kamerlingh Onnes (1914), Stanley (1997)
"""

import argparse
from math import comb, factorial, prod
from itertools import product as iprod


# ── Core formula ─────────────────────────────────────────────────────────────

def surjective_count(types: list[tuple[int, bool]], n: int) -> int:
    """
    Count unique surjective assignments of M resources onto n targets.

    Parameters
    ----------
    types : list of (m_i, is_distinguishable) tuples
        m_i               : number of resources of type i  (>= 0)
        is_distinguishable: True  → resources within type i are distinct
                            False → resources within type i are identical
    n : int
        Number of distinguishable targets (>= 1).

    Returns
    -------
    int
        Number of unique surjective assignments.
        Returns 0 when M < n (infeasible).

    Examples
    --------
    >>> surjective_count([(4, True)], 2)          # all-dist: 2!·S(4,2)
    14
    >>> surjective_count([(4, False)], 2)          # all-indist: C(3,1)
    3
    >>> surjective_count([(3, True), (2, False)], 3)   # mixed
    93
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not types:
        raise ValueError("types list must be non-empty")
    if any(m < 0 for m, _ in types):
        raise ValueError("all type sizes m_i must be >= 0")

    M   = sum(m for m, _ in types)
    D   = [m for m, d in types if     d]   # distinguishable type sizes
    I   = [m for m, d in types if not d]   # indistinguishable type sizes
    M_D = sum(D)

    # ── Boundary: impossible ────────────────────────────────────────────────
    if M < n:
        return 0

    # ── M = n: bijection closed form ────────────────────────────────────────
    #   Omega = n! / prod_{i in I} m_i!
    if M == n:
        return factorial(n) // (prod(factorial(m) for m in I) if I else 1)

    # ── M > n: general PIE formula ──────────────────────────────────────────
    #   Omega = sum_{j=0}^{n} (-1)^j C(n,j) (n-j)^{M_D}
    #                          * prod_{i in I} C((n-j)+m_i-1, m_i)
    total = 0
    for j in range(n + 1):
        t    = n - j
        sign = (-1) ** j

        # Distinguishable factor: t^{M_D}
        # (when M_D=0 this is 1; when t=0 and M_D>0 this is 0)
        dist_factor = pow(t, M_D)

        # Indistinguishable factor: stars-and-bars per type
        # C(t + m_i - 1, m_i)  →  0 when t=0 and m_i>0
        indist_factor = prod(comb(t + m - 1, m) for m in I) if I else 1

        total += sign * comb(n, j) * dist_factor * indist_factor

    return total


# ── Stirling number of the second kind ───────────────────────────────────────

def stirling2(M: int, n: int) -> int:
    """
    Stirling number of the second kind  S(M, n).

    Counts the number of ways to partition a set of M distinguishable
    objects into exactly n non-empty, unlabeled subsets.

    Relationship to surjections:
        n! * S(M, n)  =  number of surjective functions from M distinct
                         objects onto n distinct targets.

    Recurrence:  S(M, n) = n·S(M-1, n) + S(M-1, n-1)
    Boundary:    S(0, 0) = 1,  S(M, 0) = 0 for M > 0

    Examples
    --------
    >>> stirling2(4, 2)
    7
    >>> stirling2(5, 3)
    25
    """
    if M == 0 and n == 0:
        return 1
    if n == 0 or M < n:
        return 0
    return surjective_count([(M, True)], n) // factorial(n)


# ── Brute-force verifier (small inputs only) ──────────────────────────────────

def brute_force_verify(types: list[tuple[int, bool]], n: int) -> int:
    """
    Brute-force count for correctness verification on small inputs.

    Enumerates every possible assignment for each type, then counts
    combinations that cover all n targets.  Complexity is exponential
    in M — use only for small M and n (M <= ~8, n <= ~4).

    For distinguishable type i: enumerates all n^{m_i} target-index tuples.
    For indistinguishable type i: enumerates all stars-and-bars distributions
        (tuples of n non-negative counts summing to m_i).
    """
    # Build the space of assignments for each type
    type_spaces = []
    for m, is_dist in types:
        if is_dist:
            # Each of m distinct resources independently picks a target
            type_spaces.append(list(iprod(range(n), repeat=m)))
        else:
            # m identical resources → (count per target), sum = m
            type_spaces.append([
                counts
                for counts in iprod(range(m + 1), repeat=n)
                if sum(counts) == m
            ])

    count = 0
    for combo in iprod(*type_spaces):
        # Determine which targets are covered across all types
        covered = set()
        for i, (_, is_dist) in enumerate(types):
            if is_dist:
                covered.update(combo[i])                          # target indices
            else:
                covered.update(t for t, c in enumerate(combo[i]) if c > 0)

        if len(covered) == n:   # surjective
            count += 1

    return count


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_type_spec(spec: str) -> tuple[int, bool]:
    """
    Parse a resource-type spec into (m_i, is_distinguishable).

    Accepted forms: 4:D, 4:d, 4D, 4d, 4:I, 4:i, 4I, 4i
    """
    spec = spec.strip()
    if not spec:
        raise argparse.ArgumentTypeError("type spec must not be empty")

    if ":" in spec:
        count_str, kind = spec.split(":", 1)
        kind = kind.strip().upper()
    else:
        count_str, kind = spec[:-1], spec[-1].upper()

    if kind not in {"D", "I"}:
        raise argparse.ArgumentTypeError(
            f"invalid type kind {kind!r} in {spec!r}; use D (distinguishable) or I (indistinguishable)"
        )

    try:
        count = int(count_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid resource count {count_str!r} in {spec!r}"
        ) from exc

    if count < 0:
        raise argparse.ArgumentTypeError(
            f"resource count must be >= 0, got {count} in {spec!r}"
        )

    return count, kind == "D"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique surjective assignments of M resources "
            "(mixed distinguishable/indistinguishable types) onto n targets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -n 2 --type 4:D\n"
            "  %(prog)s -n 3 --type 3:D --type 2:I\n"
            "  %(prog)s -n 3 --type 2I --type 3I --verify\n"
            "  %(prog)s --demo"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in demo and verification suite",
    )
    parser.add_argument(
        "-n",
        "--targets",
        type=int,
        metavar="N",
        help="number of distinguishable targets (required unless --demo is used)",
    )
    parser.add_argument(
        "--type",
        action="append",
        dest="types",
        metavar="SPEC",
        type=_parse_type_spec,
        help=(
            "resource type as COUNT:KIND, where KIND is D (distinguishable) "
            "or I (indistinguishable); repeatable (e.g. --type 4:D --type 2:I)"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also compute the brute-force count for small inputs",
    )
    return parser


def _run_cli(args: argparse.Namespace) -> int:
    if args.demo:
        _run_demo()
        return 0

    if args.targets is None:
        raise SystemExit("error: -n/--targets is required unless --demo is used")
    if not args.types:
        raise SystemExit("error: at least one --type SPEC is required unless --demo is used")

    result = surjective_count(args.types, args.targets)
    types_str = ", ".join(f"({m}, {'D' if d else 'I'})" for m, d in args.types)
    print(f"surjective_count([{types_str}], n={args.targets})")
    print(f"-> {result:,}")

    if args.verify:
        brute = brute_force_verify(args.types, args.targets)
        match = "OK" if brute == result else "FAIL"
        print(f"brute-force: {brute:,}  {match}")

    return 0


# ── Demo & verification ───────────────────────────────────────────────────────

def _run_demo():
    SEP = "─" * 68

    print("=" * 68)
    print("  SURJECTIVE ASSIGNMENT COUNTER")
    print("  Formula vs brute-force verification")
    print("=" * 68)

    # Each entry: (description, types, n, expected_or_None)
    cases = [
        # ── Single distinguishable type ──────────────────────────────────
        ("M=4 all-dist,   n=2  [2!·S(4,2)=14]",
         [(4, True)], 2, 14),

        ("M=5 all-dist,   n=3  [3!·S(5,3)=150]",
         [(5, True)], 3, 150),

        # ── Single indistinguishable type ─────────────────────────────────
        ("M=4 all-indist, n=2  [C(3,1)=3]",
         [(4, False)], 2, 3),

        ("M=5 all-indist, n=3  [C(4,2)=6]",
         [(5, False)], 3, 6),

        ("M=6 all-indist, n=4  [C(5,3)=10]",
         [(6, False)], 4, 10),

        # ── M = n (bijection) ────────────────────────────────────────────
        ("M=n=4, all-dist             [4!=24]",
         [(1,True),(1,True),(1,True),(1,True)], 4, 24),

        ("M=n=4, all-indist 1 type    [4!/4!=1]",
         [(4, False)], 4, 1),

        ("M=n=4, mixed 2D+2I          [4!/2!=12]",
         [(2, True), (2, False)], 4, 12),

        ("M=n=6, mixed 3D+2I+1I       [6!/(2!·1!)]",
         [(3, True), (2, False), (1, False)], 6,
         factorial(6) // (factorial(2) * factorial(1))),

        # ── Mixed types, M > n ───────────────────────────────────────────
        ("M=5 (3D+2I),        n=3",
         [(3, True), (2, False)], 3, None),

        ("M=5 (2D+3I),        n=3",
         [(2, True), (3, False)], 3, None),

        ("M=6 (2D+2I+2I),     n=3",
         [(2, True), (2, False), (2, False)], 3, None),

        ("M=6 (3D+3D),        n=3  [3!·S(6,3)=1·540]",
         [(3, True), (3, True)], 3, factorial(3) * stirling2(6, 3)),

        # ── All-indistinguishable, multiple types ─────────────────────────
        ("M=6 (2I+2I+2I),     n=3",
         [(2, False), (2, False), (2, False)], 3, None),

        ("M=5 (3I+2I),        n=3",
         [(3, False), (2, False)], 3, None),

        ("M=7 (3I+2I+2I),     n=4",
         [(3, False), (2, False), (2, False)], 4, None),

        # ── Boundary ─────────────────────────────────────────────────────
        ("M=2 < n=3           [impossible → 0]",
         [(2, False)], 3, 0),
    ]

    print(f"\n{'Case':<42} {'Formula':>9} {'Brute':>7} {'OK?':>5}")
    print(SEP)

    all_pass = True
    for desc, types, n, expected in cases:
        formula = surjective_count(types, n)
        brute   = brute_force_verify(types, n) 
        match   = formula == brute
        expect_ok = (expected is None) or (formula == expected)
        ok_str  = "✓" if (match and expect_ok) else "✗ FAIL"
        if not (match and expect_ok):
            all_pass = False
        print(f"{desc:<42} {formula:>9,} {brute:>7,} {ok_str:>5}")

    print(SEP)
    status = "ALL PASSED ✓" if all_pass else "FAILURES DETECTED ✗"
    print(f"\n{status}\n")

    # ── Stirling number table ─────────────────────────────────────────────
    print("Stirling Numbers of the Second Kind  S(M, n)")
    print(SEP)
    N_MAX = 7
    header_label = "M \\ n"
    print(f"{header_label:>5}", end="")
    for n in range(1, N_MAX):
        print(f"{n:>10}", end="")
    print()
    print("─" * (5 + 10 * (N_MAX - 1)))
    for M in range(1, N_MAX):
        print(f"{M:>5}", end="")
        for n in range(1, N_MAX):
            print(f"{stirling2(M, n):>10,}", end="")
        print()
    print()

    # ── API usage examples ────────────────────────────────────────────────
    print("API usage examples")
    print(SEP)
    examples = [
        ("All-distinguishable, M=6, n=3",
         [(6, True)], 3),
        ("All-indistinguishable, M=6, n=3",
         [(6, False)], 3),
        ("Mixed: 2 dist + 2 indist, M=4, n=2",
         [(2, True), (2, False)], 2),
        ("Three indist types [2,3,1], n=3",
         [(2, False), (3, False), (1, False)], 3),
    ]
    for desc, types, n in examples:
        result = surjective_count(types, n)
        types_str = ", ".join(
            f"({m}, {'D' if d else 'I'})" for m, d in types
        )
        print(f"  surjective_count([{types_str}], n={n})")
        print(f"    {desc}")
        print(f"    → {result:,}\n")


if __name__ == "__main__":
    raise SystemExit(_run_cli(_build_arg_parser().parse_args()))

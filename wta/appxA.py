from math import comb, factorial, prod

def surjective_count(types: list[tuple[int, bool]], n: int) -> int:
    # Count unique surjective assignments of M resources onto n targets.

    # types : list of (m_i, is_distinguishable) tuples, where m_i is the number of resources of 
    # type i and is_distinguishable is a boolean indicating if the resources of type i are distinguishable.
    # n: int = number of targets (all considered distinguishable).

    # Examples:
    # surjective_count([(4, True)], 2)              # all-dist: 2!·S(4,2), where S=Stirling number of the 2nd kind
    # surjective_count([(4, False)], 2)             # all-indist: C(3,1)
    # surjective_count([(3, True), (2, False)], 3)  # mixed types, use full PIE sum

    M   = sum(m for m, _ in types)
    D   = [m for m, d in types if     d]   # distinguishable type sizes
    I   = [m for m, d in types if not d]   # indistinguishable type sizes
    M_D = sum(D)

    # M < n: insufficient resources
    # Omega = 0
    if M < n:
        return 0

    # M = n: bijection
    # Omega = n! / prod_{i in I} m_i!
    if M == n:
        return factorial(n) // (prod(factorial(m) for m in I) if I else 1)

    # M > n: general PIE formula 
    # Omega = sum_{j=0}^{n} (-1)^j C(n,j) (n-j)^{M_D}
    #                          * prod_{i in I} C((n-j)+m_i-1, m_i)
    total = 0
    for j in range(n + 1):
        t = n - j
        sign = (-1) ** j

        # Distinguishable factor: t^{M_D}
        # when M_D=0, this is 1; when t=0 and M_D>0, this is 0
        dist_factor = pow(t, M_D)

        # Indistinguishable factor: stars-and-bars per type; C(t + m_i - 1, m_i)
        # when t=0 and m_i>0, this is 0
        indist_factor = prod(comb(t + m - 1, m) for m in I) if I else 1

        # add the product of the three factors to the running sum
        total += sign * comb(n, j) * dist_factor * indist_factor 

    return total

if __name__ == "__main__":
    result_dist = surjective_count([(4, True)], 2)                  # all-dist: 2!·S(4,2), where S=Stirling number of the second kind
    result_indist = surjective_count([(4, False)], 2)               # all-indist: C(3,1) 
    result_mixed = surjective_count([(3, True), (2, False)], 3)     # mixed types, use full PIE sum
    print(result_dist)
    print(result_indist)
    print(result_mixed)

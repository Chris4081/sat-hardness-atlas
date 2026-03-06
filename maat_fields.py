import numpy as np

def harmony_field(clauses, variables):
    denom = (len(clauses) + variables)
    if denom == 0:
        return 0.0
    return 1.0 - abs(len(clauses) - variables) / denom


def balance_field(clauses):
    pos = sum(l > 0 for c in clauses for l in c)
    neg = sum(l < 0 for c in clauses for l in c)
    total = pos + neg
    if total == 0:
        return 0.0
    return 1.0 - abs(pos - neg) / total


def creativity_field(clauses, variables=None):
    if not clauses:
        return 0.0
    unique_patterns = len(set(tuple(sorted(c)) for c in clauses))
    return unique_patterns / len(clauses)


def connection_field_sparse(clauses, variables):
    """
    Connection = mittlere Anzahl unterschiedlicher Nachbarn pro Variable (0..1 skaliert via / (variables-1))
    -> ohne NxN Matrix
    """
    if variables <= 1:
        return 0.0

    neighbors = [set() for _ in range(variables)]
    for clause in clauses:
        vars_in_clause = [abs(l)-1 for l in clause]
        # unique in clause, damit keine Selbst-/Doppelzählung
        vars_in_clause = list(set(v for v in vars_in_clause if 0 <= v < variables))
        for i in vars_in_clause:
            for j in vars_in_clause:
                if i != j:
                    neighbors[i].add(j)

    avg_deg = np.mean([len(n) for n in neighbors])
    return float(avg_deg / (variables - 1))


def respect_field_giant_component(clauses, variables):
    if variables <= 0:
        return 0.0

    parent = list(range(variables))
    size = [1] * variables

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for clause in clauses:
        vs = [abs(l)-1 for l in clause]
        vs = list(set(v for v in vs if 0 <= v < variables))
        if len(vs) >= 2:
            base = vs[0]
            for v in vs[1:]:
                union(base, v)

    comp_sizes = {}
    for i in range(variables):
        r = find(i)
        comp_sizes[r] = comp_sizes.get(r, 0) + 1

    largest = max(comp_sizes.values()) if comp_sizes else 0
    return float(largest / variables)

def respect_field_usage_fairness(clauses, variables):
    """
    R: 'Respekt' als Fairness der Variablen-Nutzung.
    1.0 = alle Variablen gleich oft genutzt
    kleiner = einige Variablen dominieren (unfair)
    """
    if variables <= 0:
        return 0.0

    counts = np.zeros(variables, dtype=float)
    for clause in clauses:
        for lit in clause:
            idx = abs(lit) - 1
            if 0 <= idx < variables:
                counts[idx] += 1

    mean = counts.mean()
    if mean <= 0:
        return 0.0

    cv = counts.std() / mean  # coefficient of variation
    return 1.0 / (1.0 + cv)
    
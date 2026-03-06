import random

def generate_sat_instance(n_vars, n_clauses, k=3):

    clauses = []

    for _ in range(n_clauses):

        clause = []

        for _ in range(k):

            var = random.randint(1, n_vars)
            sign = random.choice([-1, 1])

            clause.append(sign * var)

        clauses.append(clause)

    return clauses
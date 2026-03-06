import random

from maat_fields import (
    harmony_field,
    balance_field,
    creativity_field,
    connection_field_sparse,
    respect_field_usage_fairness
)


def random_cnf(variables=20, clauses=40, k=3):
    cnf = []
    for _ in range(clauses):

        vars_in_clause = random.sample(range(1, variables+1), k)

        clause = []
        for v in vars_in_clause:
            if random.random() < 0.5:
                v = -v
            clause.append(v)

        cnf.append(clause)

    return cnf


def test():

    variables = 30
    clauses = random_cnf(variables, 80)

    H = harmony_field(clauses, variables)
    B = balance_field(clauses)
    S = creativity_field(clauses, variables)
    V = connection_field_sparse(clauses, variables)
    R = respect_field_usage_fairness(clauses, variables)

    print("\nMAAT Fields Test\n")
    print("H (Harmony):", H)
    print("B (Balance):", B)
    print("S (Creativity):", S)
    print("V (Connection):", V)
    print("R (Respect):", R)


print("\nRespect Field Variance Test\n")

for i in range(10):

    clauses = random_cnf(30, 80)

    r = respect_field_usage_fairness(clauses, 30)

    print("Run", i, "R =", r)

rs = []
for i in range(10):
    clauses = random_cnf(30, 80)
    r = respect_field_usage_fairness(clauses, 30)
    rs.append(r)
    print("Run", i, "R =", r)

print("R mean =", sum(rs)/len(rs))
print("R min  =", min(rs), "max =", max(rs))

if __name__ == "__main__":
    test()
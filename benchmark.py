# benchmark.py
import pandas as pd

from maat_fields import (
    harmony_field,
    balance_field,
    creativity_field,
    connection_field_sparse,
    respect_field_usage_fairness,
)
from complexity import maat_score
from generator import generate_sat_instance
from solver import solve_sat


def run_benchmark(samples: int, n_vars: int, alpha: float) -> pd.DataFrame:
    rows = []
    n_clauses = int(alpha * n_vars)

    for _ in range(samples):
        clauses = generate_sat_instance(n_vars, n_clauses)

        H = harmony_field(clauses, n_vars)
        B = balance_field(clauses)
        S = creativity_field(clauses, n_vars)
        V = connection_field_sparse(clauses, n_vars)
        R = respect_field_usage_fairness(clauses, n_vars)

        C_hat = maat_score(H, B, S, V, R)
        result, runtime = solve_sat(clauses)

        rows.append({
            "H": H, "B": B, "S": S, "V": V, "R": R,
            "C_hat": C_hat,
            "runtime": runtime,
            "sat": bool(result),
        })

    df = pd.DataFrame(rows)

    print("R unique:", sorted(df["R"].unique())[:10], "count:", df["R"].nunique())
    print("R mean:", df["R"].mean(), "min:", df["R"].min(), "max:", df["R"].max())

    return df
import time
from pysat.solvers import Minisat22

def solve_sat(clauses):

    start = time.time()

    solver = Minisat22()

    for c in clauses:
        solver.add_clause(c)

    result = solver.solve()

    runtime = time.time() - start

    solver.delete()

    return result, runtime
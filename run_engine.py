from benchmark import run_benchmark
from visualization import plot_maat_vs_runtime

samples = 300
variables = 100

alphas = [3.0,3.4,3.8,4.0,4.1,4.2,4.26,4.4,4.6,5.0]

for alpha in alphas:

    print("running alpha:", alpha)

    df = run_benchmark(samples, variables, alpha)

    plot_maat_vs_runtime(df)
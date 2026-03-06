# engine.py
import os
import pandas as pd

from config import CONFIG
from benchmark import run_benchmark
from visualization import plot_maat_vs_runtime
from heatmap import plot_sat_hardness_heatmap
from phase_transition import plot_sat_probability
from maat_predictor import plot_chat_vs_runtime
from maat_predictor import plot_sat_prob_vs_chat
from maat_regression import run_regression
from landscape import plot_hardness_landscape
from sat_hardness_curve import plot_median_runtime_vs_alpha
from hardness_predictor import run_hardness_predictor


def alpha_tag(alpha: float) -> str:
    # 3.0 -> "3p0" damit Dateinamen sauber bleiben
    return f"{alpha}".replace(".", "p")


class MaatCosmosEngine:
    def __init__(self):
        self.samples = CONFIG["samples"]
        self.variables = CONFIG["variables"]
        self.alphas = CONFIG["alphas"]
        self.output_dir = CONFIG["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print("🌌 MAAT COSMOS ENGINE v6")
        print("starting benchmark...\n")

        all_results = []

        for alpha in self.alphas:
            print(f"running alpha = {alpha}")

            df = run_benchmark(
                samples=self.samples,
                n_vars=self.variables,
                alpha=float(alpha),
            )

            df["alpha"] = float(alpha)
            all_results.append(df)

            tag = alpha_tag(alpha)
            path = os.path.join(self.output_dir, f"results_alpha_{tag}.csv")
            df.to_csv(path, index=False)
            print("saved:", path)

        final_df = pd.concat(all_results, ignore_index=True)

        final_path = os.path.join(self.output_dir, "maat_cosmos_full_results.csv")
        final_df.to_csv(final_path, index=False)
        print("\nall results saved →", final_path)

        print("\ncreating visualization...")
        plot_maat_vs_runtime(final_df, outdir=self.output_dir)

        print("\ncreating hardness heatmap...")
        plot_sat_hardness_heatmap(final_df, outdir=self.output_dir)
        
        print("\ncreating phase transition plot...")
        plot_sat_probability(final_df, outdir=self.output_dir)
        
        print("\ncreating C_hat predictor plots...")
        plot_chat_vs_runtime(final_df, outdir=self.output_dir)
        plot_sat_prob_vs_chat(final_df, outdir=self.output_dir)

        print("\nrunning regression analysis...")
        run_regression(final_df)

        print("\ncreating hardness landscape...")
        plot_hardness_landscape(final_df, outdir=self.output_dir)

        print("\ncreating SAT hardness curve...")
        plot_median_runtime_vs_alpha(final_df, outdir=self.output_dir)

        print("\nrunning hardness predictor...")
        run_hardness_predictor(final_df)

        print("\n✅ MAAT COSMOS ENGINE finished")


if __name__ == "__main__":
    engine = MaatCosmosEngine()
    engine.run()
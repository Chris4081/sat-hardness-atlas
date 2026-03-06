# landscape.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hardness_landscape(
    df: pd.DataFrame,
    outdir: str = "results",
    bins_alpha: int = 12,
    bins_chat: int = 18,
    use_log_runtime: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    d = df.copy()

    # Safe runtime transform
    rt = d["runtime"].astype(float).values
    if use_log_runtime:
        z = np.log10(np.maximum(rt, 1e-12))
        z_label = "log10(runtime)"
        fname_tag = "log"
    else:
        z = rt
        z_label = "runtime"
        fname_tag = "raw"

    d["_z"] = z

    # Bin edges
    alpha_vals = d["alpha"].astype(float).values
    chat_vals = d["C_hat"].astype(float).values

    a_edges = np.linspace(alpha_vals.min(), alpha_vals.max(), bins_alpha + 1)
    c_edges = np.linspace(chat_vals.min(), chat_vals.max(), bins_chat + 1)

    # Assign bins
    d["a_bin"] = np.digitize(alpha_vals, a_edges) - 1
    d["c_bin"] = np.digitize(chat_vals, c_edges) - 1

    # Clamp in range
    d["a_bin"] = d["a_bin"].clip(0, bins_alpha - 1)
    d["c_bin"] = d["c_bin"].clip(0, bins_chat - 1)

    # Pivot -> mean z per cell
    grid = (
        d.groupby(["c_bin", "a_bin"])["_z"]
        .mean()
        .unstack("a_bin")
        .reindex(index=range(bins_chat), columns=range(bins_alpha))
    )

    Z = grid.values  # shape (bins_chat, bins_alpha)

    # Centers for axes
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])

    # ---------- 2D HEATMAP ----------
    plt.figure()
    # imshow expects [row, col] => [c_bin, a_bin]
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[a_edges[0], a_edges[-1], c_edges[0], c_edges[-1]],
    )
    plt.colorbar(im, label=f"Mean {z_label}")
    plt.xlabel("alpha (clauses / variables)")
    plt.ylabel("C_hat")
    plt.title(f"Hardness Landscape (mean {z_label})")

    out_heat = os.path.join(outdir, f"hardness_landscape_heatmap_{fname_tag}.png")
    plt.tight_layout()
    plt.savefig(out_heat, dpi=200)
    plt.close()
    print("landscape heatmap saved:", out_heat)

    # ---------- 3D SURFACE ----------
    # only if matplotlib 3D available (it is by default)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    A, C = np.meshgrid(a_centers, c_centers)  # shape (bins_chat, bins_alpha)

    # Mask NaNs so empty bins don't create weird spikes
    Z_masked = np.array(Z, dtype=float)
    Z_masked[np.isnan(Z_masked)] = np.nan

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, C, np.nan_to_num(Z_masked, nan=np.nanmin(Z_masked[np.isfinite(Z_masked)])),
                    rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("C_hat")
    ax.set_zlabel(z_label)
    ax.set_title(f"Hardness Landscape Surface ({z_label})")

    out_surf = os.path.join(outdir, f"hardness_landscape_surface_{fname_tag}.png")
    plt.tight_layout()
    plt.savefig(out_surf, dpi=200)
    plt.close()
    print("landscape surface saved:", out_surf)

    # Optional: export the grid as CSV (handy for analysis)
    out_csv = os.path.join(outdir, f"hardness_landscape_grid_{fname_tag}.csv")
    grid.to_csv(out_csv, index=True)
    print("landscape grid saved:", out_csv)
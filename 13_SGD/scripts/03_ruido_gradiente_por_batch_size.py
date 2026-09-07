#!/usr/bin/env python3
"""Mede a dispersão do gradiente sem atualizar os parâmetros congelados."""
import argparse
import numpy as np

from sgd_video_experiments.common import configure_matplotlib, ensure_dir, write_csv, write_json
from sgd_video_experiments.regression import make_linear_data, mse_loss_and_gradients, sample_batch_gradients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/03_ruido_gradiente")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 200])
    parser.add_argument("--w", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.0)
    args = parser.parse_args()

    configure_matplotlib(); import matplotlib.pyplot as plt
    out = ensure_dir(args.output_dir)
    data = make_linear_data(args.samples, seed=args.seed)
    full_loss, full_dw, full_db = mse_loss_and_gradients(data.x, data.y, args.w, args.b)
    draws_rows, summaries = [], []
    fig, ax = plt.subplots(figsize=(7, 6))
    for batch_size in args.batch_sizes:
        gradients = sample_batch_gradients(data.x, data.y, args.w, args.b, batch_size, args.draws, args.seed + batch_size)
        distances = np.linalg.norm(gradients - np.array([full_dw, full_db]), axis=1)
        for index, ((dw, db), distance) in enumerate(zip(gradients, distances)):
            draws_rows.append({"batch_size": batch_size, "draw": index, "dw": dw, "db": db, "distance_to_full": distance})
        summaries.append({"batch_size": batch_size, "draws": args.draws, "mean_dw": gradients[:, 0].mean(), "mean_db": gradients[:, 1].mean(), "std_dw": gradients[:, 0].std(), "std_db": gradients[:, 1].std(), "mean_distance_to_full": distances.mean()})
        shown = gradients[:min(300, len(gradients))]
        ax.scatter(shown[:, 0], shown[:, 1], s=12, alpha=0.28, label=f"batch={batch_size}")
    ax.scatter([full_dw], [full_db], marker="*", s=220, color="black", label="full batch", zorder=10)
    ax.set(xlabel="dw", ylabel="db", title=f"Gradientes em w={args.w:g}, b={args.b:g} (sem updates)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "dispersao_gradientes.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    batch_sizes = [row["batch_size"] for row in summaries]
    mean_distances = [row["mean_distance_to_full"] for row in summaries]
    ax.plot(batch_sizes, mean_distances, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="batch size (escala log2)",
        ylabel="distância média até o gradiente full batch",
        title="Batches maiores reduzem a dispersão do gradiente",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "ruido_por_batch_size.png", dpi=160); plt.close(fig)

    write_csv(out / "gradientes_sorteados.csv", draws_rows); write_csv(out / "resumo_por_batch.csv", summaries)
    write_json(out / "metricas.json", {"seed": args.seed, "frozen_w": args.w, "frozen_b": args.b, "full_batch": {"loss": full_loss, "dw": full_dw, "db": full_db}, "batch_summaries": summaries})
    print(f"full batch: dw={full_dw:.5f}, db={full_db:.5f}")
    for row in summaries: print(f"batch={row['batch_size']}: distância média={row['mean_distance_to_full']:.5f}")


if __name__ == "__main__":
    main()

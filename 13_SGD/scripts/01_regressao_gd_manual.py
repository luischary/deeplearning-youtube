#!/usr/bin/env python3
"""Treina y = wx + b com gradientes escritos explicitamente."""
import argparse

from sgd_video_experiments.common import configure_matplotlib, ensure_dir, write_csv, write_json
from sgd_video_experiments.regression import make_linear_data, run_gradient_descent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/01_regressao_gd_manual")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--initial-w", type=float, default=-1.0)
    parser.add_argument("--initial-b", type=float, default=0.0)
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt
    out = ensure_dir(args.output_dir)
    data = make_linear_data(args.samples, args.noise_std, args.seed)
    trajectory = run_gradient_descent(data.x, data.y, args.learning_rate, args.steps, args.initial_w, args.initial_b)
    final = trajectory[-1]
    write_csv(out / "trajetoria.csv", trajectory)
    write_json(out / "metricas.json", {**vars(args), "final": final, "true_w": 2.0, "true_b": 3.0})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(data.x, data.y, s=18, alpha=0.65, label="dados")
    x_line = sorted(data.x)
    axes[0].plot(x_line, [final["w"] * x + final["b"] for x in x_line], color="tab:red", label="reta final")
    axes[0].plot(x_line, [2 * x + 3 for x in x_line], "--", color="black", label="reta geradora")
    axes[0].set(xlabel="x", ylabel="y", title="Regressão linear por GD manual")
    axes[0].legend()
    axes[1].plot([row["step"] for row in trajectory], [row["loss"] for row in trajectory])
    axes[1].set(xlabel="passo", ylabel="MSE", title="Loss por atualização")
    axes[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(out / "regressao_e_loss.png", dpi=160); plt.close(fig)
    print(f"final: loss={final['loss']:.6f}, w={final['w']:.4f}, b={final['b']:.4f}; arquivos em {out}")


if __name__ == "__main__":
    main()

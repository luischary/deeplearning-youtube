#!/usr/bin/env python3
"""Compara learning rates usando exatamente os mesmos dados e w,b iniciais."""
import argparse
import math

from sgd_video_experiments.common import configure_matplotlib, ensure_dir, write_csv, write_json
from sgd_video_experiments.regression import make_linear_data, run_gradient_descent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/02_regressao_learning_rates")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.005, 0.05, 0.9])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--initial-w", type=float, default=-1.0)
    parser.add_argument("--initial-b", type=float, default=0.0)
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt
    out = ensure_dir(args.output_dir)
    data = make_linear_data(args.samples, seed=args.seed)
    all_rows, summary = [], []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in args.learning_rates:
        trajectory = run_gradient_descent(data.x, data.y, lr, args.steps, args.initial_w, args.initial_b)
        for row in trajectory:
            all_rows.append({"learning_rate": lr, **row})
        final = trajectory[-1]
        finite = math.isfinite(float(final["loss"]))
        summary.append({"learning_rate": lr, "final_loss": final["loss"], "final_w": final["w"], "final_b": final["b"], "finite": finite})
        losses = [min(float(r["loss"]), 1e100) for r in trajectory]
        ax.plot(range(len(losses)), losses, label=f"lr={lr:g}")
    write_csv(out / "trajetorias.csv", all_rows)
    write_csv(out / "resumo.csv", summary)
    write_json(out / "metricas.json", {"seed": args.seed, "steps": args.steps, "initial_w": args.initial_w, "initial_b": args.initial_b, "results": summary})
    ax.set(xlabel="passo", ylabel="MSE", title="Mesmo início, learning rates diferentes"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparacao_learning_rates.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in args.learning_rates:
        rows = [row for row in all_rows if row["learning_rate"] == lr]
        ax.plot([row["step"] for row in rows], [row["loss"] for row in rows], label=f"lr={lr:g}")
    initial_loss = float(all_rows[0]["loss"])
    ax.set_ylim(bottom=max(initial_loss * 0.005, 1e-3), top=initial_loss * 1.25)
    ax.set_yscale("log")
    ax.set(xlabel="passo", ylabel="MSE", title="Zoom: progresso antes da divergência sair da escala")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "comparacao_learning_rates_zoom.png", dpi=160); plt.close(fig)
    for row in summary:
        print(f"lr={row['learning_rate']:g}: loss={row['final_loss']:.6g}, w={row['final_w']:.4g}, b={row['final_b']:.4g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compara learning rates no MNIST com inicialização e ordem de dados idênticas."""

import argparse

from sgd_video_experiments.common import (
    configure_matplotlib,
    ensure_dir,
    seed_everything,
    write_csv,
    write_json,
)
from sgd_video_experiments.mnist import (
    initial_state,
    make_loaders,
    model_from_state,
    train_sgd,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/05_mnist_learning_rates")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.01, 0.1, 0.5, 1.0]
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.limit_train = args.limit_train or 2000
        args.limit_test = args.limit_test or 500
        args.epochs = min(args.epochs, 1)

    configure_matplotlib()
    import matplotlib.pyplot as plt

    seed_everything(args.seed)
    out = ensure_dir(args.output_dir)
    state = initial_state(args.seed, args.hidden_size)
    all_updates, all_epochs, summary = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for lr in args.learning_rates:
        # Recriar loader reinicia o gerador: todos veem a mesma ordem de mini-batches.
        train_loader, test_loader = make_loaders(
            args.data_dir, args.batch_size, args.seed, args.limit_train, args.limit_test
        )
        model = model_from_state(state, args.hidden_size)
        updates, epochs = train_sgd(
            model, train_loader, test_loader, lr, args.epochs, args.device
        )
        all_updates.extend({"learning_rate": lr, **row} for row in updates)
        all_epochs.extend({"learning_rate": lr, **row} for row in epochs)
        final = epochs[-1]
        summary.append({"learning_rate": lr, **final})
        batch_losses = [r["batch_loss"] for r in updates]
        moving_window = min(50, len(batch_losses))
        moving_average = [
            sum(batch_losses[max(0, i - moving_window + 1) : i + 1])
            / min(i + 1, moving_window)
            for i in range(len(batch_losses))
        ]
        axes[0].plot(
            [r["samples_seen"] for r in updates], moving_average, label=f"lr={lr:g}"
        )
        axes[1].plot(
            [r["epoch"] for r in epochs],
            [r["test_loss_per_sample"] for r in epochs],
            marker="o",
            label=f"lr={lr:g}",
        )
        axes[2].plot(
            [r["epoch"] for r in epochs],
            [100 * r["test_accuracy"] for r in epochs],
            marker="o",
            label=f"lr={lr:g}",
        )
    write_csv(out / "loss_por_update.csv", all_updates)
    write_csv(out / "metricas_por_epoca.csv", all_epochs)
    write_csv(out / "resumo.csv", summary)
    write_json(out / "metricas.json", {**vars(args), "results": summary})
    axes[0].set(
        xlabel="amostras processadas",
        ylabel="loss (média móvel de 50 batches)",
        title="Loss durante o treino",
    )
    axes[1].set(xlabel="época", ylabel="loss de teste", title="Generalização: loss")
    axes[2].set(
        xlabel="época", ylabel="acurácia de teste (%)", title="Generalização: acurácia"
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("MNIST: mesmos dados e inicialização, learning rates diferentes")
    fig.tight_layout()
    fig.savefig(out / "comparacao_learning_rates.png", dpi=160)
    plt.close(fig)
    for row in summary:
        print(
            f"lr={row['learning_rate']:g}: loss={row['test_loss_per_sample']:.4f}, accuracy={row['test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()

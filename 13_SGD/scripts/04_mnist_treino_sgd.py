#!/usr/bin/env python3
"""Treina uma MLP pequena no MNIST com mini-batch e torch.optim.SGD."""
import argparse
from pathlib import Path

from sgd_video_experiments.common import configure_matplotlib, ensure_dir, seed_everything, write_csv, write_json
from sgd_video_experiments.mnist import build_model, make_loaders, train_sgd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/04_mnist_treino")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--quick", action="store_true", help="usa 2.000/500 amostras e no máximo 1 época")
    args = parser.parse_args()
    if args.quick:
        args.limit_train = args.limit_train or 2000; args.limit_test = args.limit_test or 500; args.epochs = min(args.epochs, 1)

    configure_matplotlib(); import matplotlib.pyplot as plt; import torch
    seed_everything(args.seed)
    out = ensure_dir(args.output_dir)
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.seed, args.limit_train, args.limit_test)
    model = build_model(args.hidden_size)
    updates, epochs = train_sgd(model, train_loader, test_loader, args.learning_rate, args.epochs, args.device)
    write_csv(out / "loss_por_update.csv", updates); write_csv(out / "metricas_por_epoca.csv", epochs)
    final = epochs[-1]
    write_json(out / "metricas.json", {**vars(args), "final": final})
    torch.save({"model_state": model.state_dict(), "hidden_size": args.hidden_size, "seed": args.seed, "metrics": final}, out / "checkpoint.pt")
    batch_losses = [r["batch_loss"] for r in updates]
    moving_window = min(50, len(batch_losses))
    moving_average = [sum(batch_losses[max(0, i - moving_window + 1):i + 1]) / min(i + 1, moving_window) for i in range(len(batch_losses))]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot([r["update"] for r in updates], batch_losses, alpha=.35, label="loss do mini-batch"); ax.plot([r["update"] for r in updates], moving_average, linewidth=2, label=f"média móvel ({moving_window} batches)")
    ax.set(xlabel="atualização", ylabel="CrossEntropyLoss", title="MNIST: loss por update"); ax.legend(); fig.tight_layout(); fig.savefig(out / "loss_por_update.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epoch_numbers = [r["epoch"] for r in epochs]
    axes[0].plot(epoch_numbers, [r["train_loss_per_sample"] for r in epochs], marker="o", label="treino")
    axes[0].plot(epoch_numbers, [r["test_loss_per_sample"] for r in epochs], marker="o", label="teste")
    axes[0].set(xlabel="época", ylabel="loss média por amostra", title="Loss de treino e teste")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(epoch_numbers, [100 * r["test_accuracy"] for r in epochs], marker="o", color="tab:green")
    axes[1].set(xlabel="época", ylabel="acurácia de teste (%)", title="Acurácia no conjunto de teste")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"MNIST com SGD — lr={args.learning_rate:g}, batch={args.batch_size}")
    fig.tight_layout(); fig.savefig(out / "curvas_por_epoca.png", dpi=160); plt.close(fig)

    print(f"final: test_loss={final['test_loss_per_sample']:.4f}, accuracy={final['test_accuracy']:.2%}, updates={final['updates']}; arquivos em {out}")


if __name__ == "__main__": main()

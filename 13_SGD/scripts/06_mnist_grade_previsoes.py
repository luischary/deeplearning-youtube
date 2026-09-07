#!/usr/bin/env python3
"""Gera uma pequena grade de previsões a partir do checkpoint do script 04."""
import argparse
from pathlib import Path

from sgd_video_experiments.common import configure_matplotlib, ensure_dir, seed_everything, write_csv, write_json
from sgd_video_experiments.mnist import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/06_mnist_previsoes")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="outputs/04_mnist_treino/checkpoint.pt")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    configure_matplotlib(); import matplotlib.pyplot as plt; import torch
    from torchvision import datasets, transforms
    seed_everything(args.seed); out = ensure_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {checkpoint_path}. Execute primeiro o script 04.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(int(payload["hidden_size"])); model.load_state_dict(payload["model_state"]); model.eval()
    dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    generator = torch.Generator().manual_seed(args.seed); indices = torch.randperm(len(dataset), generator=generator)[:args.count].tolist()
    images = torch.stack([dataset[i][0] for i in indices]); labels = torch.tensor([dataset[i][1] for i in indices])
    with torch.no_grad(): probabilities = model(images).softmax(dim=1); confidence, predictions = probabilities.max(dim=1)
    rows = [{"index": i, "target": int(target), "prediction": int(pred), "confidence": float(conf), "correct": bool(target == pred)} for i, target, pred, conf in zip(indices, labels, predictions, confidence)]
    write_csv(out / "previsoes.csv", rows); write_json(out / "metricas.json", {"seed": args.seed, "checkpoint": str(checkpoint_path), "count": len(rows), "accuracy_in_grid": sum(r["correct"] for r in rows) / len(rows)})
    columns = 4
    lines = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(8, 2 * lines), squeeze=False)
    flat_axes = axes.ravel()
    for ax, image, row in zip(flat_axes, images, rows):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"real {row['target']} | pred {row['prediction']}\n{row['confidence']:.0%}", color="green" if row["correct"] else "red", fontsize=9)
        ax.axis("off")
    for ax in flat_axes[len(rows):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "grade_previsoes.png", dpi=160); plt.close(fig)
    print(f"grade: {sum(r['correct'] for r in rows)}/{len(rows)} corretas; arquivos em {out}")


if __name__ == "__main__": main()

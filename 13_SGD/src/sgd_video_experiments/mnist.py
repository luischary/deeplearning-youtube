from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .common import seed_everything


def build_model(hidden_size: int = 64):
    import torch.nn as nn

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )


def make_loaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    limit_train: int | None = None,
    limit_test: int | None = None,
):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    if limit_train is not None:
        train_data = Subset(train_data, range(min(limit_train, len(train_data))))
    if limit_test is not None:
        test_data = Subset(test_data, range(min(limit_test, len(test_data))))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def evaluate(model, loader, device: str = "cpu") -> dict[str, float | int]:
    import torch

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += labels.numel()
    return {
        "loss_per_sample": total_loss / count,
        "accuracy": correct / count,
        "samples": count,
    }


def train_sgd(
    model,
    train_loader,
    test_loader,
    learning_rate: float,
    epochs: int,
    device: str = "cpu",
):
    import torch

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    updates: list[dict[str, float | int]] = []
    epochs_rows: list[dict[str, float | int]] = []
    update, samples_seen = 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_sum, epoch_samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            update += 1
            batch_n = labels.numel()
            samples_seen += batch_n
            epoch_samples += batch_n
            running_sum += float(loss.item()) * batch_n
            updates.append(
                {
                    "epoch": epoch,
                    "update": update,
                    "samples_seen": samples_seen,
                    "batch_size": batch_n,
                    "batch_loss": float(loss.item()),
                    "running_loss_per_sample": running_sum / epoch_samples,
                }
            )
        metrics = evaluate(model, test_loader, device)
        epochs_rows.append(
            {
                "epoch": epoch,
                "updates": update,
                "samples_seen": samples_seen,
                "train_loss_per_sample": running_sum / epoch_samples,
                "test_loss_per_sample": metrics["loss_per_sample"],
                "test_accuracy": metrics["accuracy"],
            }
        )
    return updates, epochs_rows


def initial_state(seed: int, hidden_size: int = 64):
    seed_everything(seed)
    return copy.deepcopy(build_model(hidden_size).state_dict())


def model_from_state(state: dict[str, Any], hidden_size: int = 64):
    model = build_model(hidden_size)
    model.load_state_dict(copy.deepcopy(state))
    return model

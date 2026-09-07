from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionData:
    x: np.ndarray
    y: np.ndarray


def make_linear_data(
    n_samples: int = 100, noise_std: float = 0.5, seed: int = 42
) -> RegressionData:
    if n_samples < 2:
        raise ValueError("n_samples deve ser >= 2")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n_samples)
    y = 2.0 * x + 3.0 + rng.normal(0.0, noise_std, n_samples)
    return RegressionData(x=x, y=y)


def mse_loss_and_gradients(
    x: np.ndarray, y: np.ndarray, w: float, b: float
) -> tuple[float, float, float]:
    error = w * x + b - y
    loss = float(np.mean(error**2))
    dw = float(2.0 * np.mean(error * x))
    db = float(2.0 * np.mean(error))
    return loss, dw, db


def run_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    steps: int,
    initial_w: float,
    initial_b: float,
) -> list[dict[str, float | int]]:
    if learning_rate <= 0 or steps < 0:
        raise ValueError("learning_rate deve ser positivo e steps não negativo")
    w, b = float(initial_w), float(initial_b)
    trajectory: list[dict[str, float | int]] = []
    for step in range(steps + 1):
        loss, dw, db = mse_loss_and_gradients(x, y, w, b)
        trajectory.append(
            {"step": step, "loss": loss, "w": w, "b": b, "dw": dw, "db": db}
        )
        if step < steps:
            w -= learning_rate * dw
            b -= learning_rate * db
    return trajectory


def sample_batch_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
    batch_size: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    if not 1 <= batch_size <= len(x):
        raise ValueError("batch_size deve estar entre 1 e o tamanho do conjunto")
    rng = np.random.default_rng(seed)
    result = np.empty((draws, 2), dtype=float)
    for index in range(draws):
        chosen = rng.choice(len(x), size=batch_size, replace=False)
        _, dw, db = mse_loss_and_gradients(x[chosen], y[chosen], w, b)
        result[index] = (dw, db)
    return result

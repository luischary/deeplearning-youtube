import matplotlib.pyplot as plt
from typing import List
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


def reta_inicial_vs_final(train_log: List[dict], dados, TRUE_W: float, TRUE_B: float):
    # plota a reta final
    initial = train_log[0]
    final = train_log[-1]
    x_line = np.linspace(dados.x.min(), dados.x.max(), 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(dados.x, dados.y, s=30, alpha=0.7)
    axes[0].plot(
        x_line,
        initial["w"] * x_line + initial["b"],
        color="tab:orange",
        linewidth=3,
    )
    axes[0].set_title(f"Antes — MSE {initial['loss']:.3f}")

    axes[1].scatter(dados.x, dados.y, s=30, alpha=0.7)
    axes[1].plot(
        x_line,
        final["w"] * x_line + final["b"],
        color="tab:green",
        linewidth=3,
    )
    axes[1].plot(
        x_line,
        TRUE_W * x_line + TRUE_B,
        "--",
        color="black",
        label="reta geradora",
    )
    axes[1].set_title(f"Depois — MSE {final['loss']:.3f}")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("reta_inicial_vs_final.png")
    plt.close(fig)


def evolucao_parametros(train_log: List[dict], TRUE_W: float, TRUE_B: float):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(
        [log["epoch"] for log in train_log],
        [log["w"] for log in train_log],
        linewidth=3,
        label="w aprendido",
    )
    axes[0].axhline(TRUE_W, linestyle="--", color="black", label="w verdadeiro")
    axes[0].set_ylabel("w")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        [log["epoch"] for log in train_log],
        [log["b"] for log in train_log],
        linewidth=3,
        color="tab:orange",
        label="b aprendido",
    )
    axes[1].axhline(TRUE_B, linestyle="--", color="black", label="b verdadeiro")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("b")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("evolucao_parametros.png")
    plt.close(fig)


def create_line_evolution(x: np.ndarray, y: np.ndarray, history: List[dict]):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y)
    (line,) = ax.plot([], [], linewidth=4, color="tab:orange")

    x_line = np.linspace(x.min(), x.max(), 200)

    def update(frame_index):
        row = history[frame_index]
        y_line = row["w"] * x_line + row["b"]
        line.set_data(x_line, y_line)
        return (line,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=1000 / 12,
    )

    writer = FFMpegWriter(fps=12, bitrate=3000)
    animation.save("evolucao_reta.mp4", writer=writer, dpi=100)
    plt.close(fig)


def create_loss_evolution(train_log: List[dict]):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(
        [log["epoch"] for log in train_log],
        [log["loss"] for log in train_log],
        linewidth=3,
        color="tab:blue",
        label="loss",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("evolucao_loss.png")
    plt.close(fig)

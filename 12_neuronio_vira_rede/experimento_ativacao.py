import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons

torch.manual_seed(42)
np.random.seed(42)

# 1. Configuração dos Dados (Problema das luas)
X_np, y_np = make_moons(n_samples=600, noise=0.10, random_state=42)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)


# 2. Definição dos Modelos
class RedeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class RedeNaoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),  # A quebra de linearidade aqui
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


modelo_lin = RedeLinear()
modelo_nonlin = RedeNaoLinear()

# Otimizadores e Função de Perda
opt_lin = optim.Adam(modelo_lin.parameters(), lr=0.02)
opt_nonlin = optim.Adam(modelo_nonlin.parameters(), lr=0.02)
criterion = nn.BCELoss()

# 3. Configuração do Grid de Fundo (Mais denso para pcolormesh ficar suave)
x_min, x_max = X_np[:, 0].min() - 0.2, X_np[:, 0].max() + 0.2
y_min, y_max = X_np[:, 1].min() - 0.2, X_np[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# 4. Preparando a Janela do Gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(
    "Evolução da Fronteira de Decisão durante o Treinamento",
    fontsize=16,
    fontweight="bold",
)


def configurar_eixo(ax, titulo):
    ax.set_title(titulo, fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])


configurar_eixo(ax1, "Rede Linear (Sem Ativação)")
configurar_eixo(ax2, "Rede Não-Linear (Com ReLU)")

# Textos sobrepostos
txt_lin = ax1.text(
    0.05,
    0.90,
    "",
    transform=ax1.transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
    fontsize=10,
    zorder=4,
)
txt_nonlin = ax2.text(
    0.05,
    0.90,
    "",
    transform=ax2.transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
    fontsize=10,
    zorder=4,
)

# Inicializa o fundo com matrizes vazias usando pcolormesh (shading='auto' lida com os limites dos blocos)
fundo_lin = ax1.pcolormesh(
    xx,
    yy,
    np.zeros_like(xx),
    cmap="RdBu",
    alpha=0.5,
    vmin=0.0,
    vmax=1.0,
    zorder=1,
    shading="auto",
)
fundo_nonlin = ax2.pcolormesh(
    xx,
    yy,
    np.zeros_like(xx),
    cmap="RdBu",
    alpha=0.5,
    vmin=0.0,
    vmax=1.0,
    zorder=1,
    shading="auto",
)

# Plotar os pontos estáticos POR CIMA do fundo (zorder=3)
ax1.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", edgecolors="k", s=25, zorder=3)
ax2.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", edgecolors="k", s=25, zorder=3)

# 5. Função de Atualização da Animação
PASSO_EPOCAS = 10


def update(frame):
    epoca_atual = frame * PASSO_EPOCAS

    # --- Passo de Treinamento: Modelo Linear ---
    for _ in range(PASSO_EPOCAS):
        opt_lin.zero_grad()
        loss_l = criterion(modelo_lin(X), y)
        loss_l.backward()
        opt_lin.step()

    # --- Passo de Treinamento: Modelo Não-Linear ---
    for _ in range(PASSO_EPOCAS):
        opt_nonlin.zero_grad()
        loss_nl = criterion(modelo_nonlin(X), y)
        loss_nl.backward()
        opt_nonlin.step()

    # Predições no Grid
    with torch.no_grad():
        preds_l = modelo_lin(grid).reshape(xx.shape).numpy()
        preds_nl = modelo_nonlin(grid).reshape(xx.shape).numpy()

    # --- Atualização Dinâmica usando set_array ---
    # O pcolormesh atualiza os dados achatados (flat) de forma extremamente performática
    fundo_lin.set_array(preds_l.ravel())
    fundo_nonlin.set_array(preds_nl.ravel())

    # Atualizar textos informativos
    txt_lin.set_text(f"Época: {epoca_atual}\nLoss: {loss_l.item():.4f}")
    txt_nonlin.set_text(f"Época: {epoca_atual}\nLoss: {loss_nl.item():.4f}")

    return [fundo_lin, fundo_nonlin, txt_lin, txt_nonlin]


# 6. Gerar e Renderizar a Animação
ani = FuncAnimation(fig, update, frames=60, interval=200, blit=False, repeat=False)
# Para salvar o arquivo mp4 final:
# ani.save("evolucao_redes_pcolormesh.mp4", fps=3, extra_args=["-vcodec", "libx264"])

plt.tight_layout()
plt.show()

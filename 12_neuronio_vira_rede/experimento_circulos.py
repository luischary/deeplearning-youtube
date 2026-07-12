import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# 1. Gerar dados não-lineares (Círculo)
X_np, y_np = make_circles(n_samples=600, noise=0.05, factor=0.5, random_state=42)

# Converter para tensores do PyTorch
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# 2. Definir os Modelos
class RedeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # Entrada 2D -> Camada Escondida
            nn.Linear(8, 1),   # Camada Escondida -> Saída
            nn.Sigmoid()       # Sigmoid na saída para classificação (0 a 1)
        )
    def forward(self, x): return self.net(x)

class RedeNaoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),         # A mágica da não-linearidade aqui!
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 3. Função de Treinamento Simples
def treinar_modelo(modelo, epochs=1000):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.03)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = modelo(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return modelo

print("Treinando rede SEM ativação...")
modelo_linear = treinar_modelo(RedeLinear())

print("Treinando rede COM ativação (ReLU)...")
modelo_nao_linear = treinar_modelo(RedeNaoLinear())

# 4. Função para Plotar a Fronteira de Decisão
def plot_decisao(modelo, ax, titulo):
    # Criar uma grade de pontos para avaliar o plano de fundo
    x_min, x_max = X_np[:, 0].min() - 0.2, X_np[:, 0].max() + 0.2
    y_min, y_max = X_np[:, 1].min() - 0.2, X_np[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = modelo(grid).reshape(xx.shape).numpy()
    
    # Plotar contorno preenchido (as "previsões" da rede no espaço)
    ax.contourf(xx, yy, preds, cmap="RdBu", alpha=0.6)
    # Plotar os pontos reais por cima
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", edgecolors="k", s=20)
    ax.set_title(titulo)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# 5. Renderizar o Resultado Lado a Lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_decisao(modelo_linear, ax1, "Rede de 2 Camadas SEM Ativação (Apenas Linear)")
plot_decisao(modelo_nao_linear, ax2, "Rede de 2 Camadas COM Ativação (ReLU)")
plt.tight_layout()
plt.show()
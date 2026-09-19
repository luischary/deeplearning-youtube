# imports
import numpy as np
import matplotlib.pyplot as plt

from metricas import (
    reta_inicial_vs_final,
    evolucao_parametros,
    create_line_evolution,
    create_loss_evolution,
)

# definicao de hyperparametros
TRUE_W = 2.0
TRUE_B = 3.0
N_SAMPLES = 100
NOISE_STD = 0.5
SEED = 42
LEARNING_RATE = 0.05
N_EPOCHS = 100

# inicializa os dados
rng = np.random.default_rng(SEED)
x = rng.uniform(-2.0, 2.0, size=N_SAMPLES)
y = TRUE_W * x + TRUE_B + rng.normal(0.0, NOISE_STD, size=N_SAMPLES)


def forward(x, w, b):
    return w * x + b


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# inicializa o modelo
w_modelo = -1.0
b_modelo = 0.0


dados = []
# SGD
for epoch in range(N_EPOCHS + 1):
    # - produzir previsoes
    previsao = forward(x, w_modelo, b_modelo)
    # - calcular a loss
    loss = mse_loss(y, previsao)
    # - calcular os gradientes
    erro = previsao - y
    grad_w = 2 * np.mean(erro * x)
    grad_b = 2 * np.mean(erro)

    dados.append(
        {
            "epoch": epoch,
            "w": w_modelo,
            "b": b_modelo,
            "loss": loss,
            "dw": grad_w,
            "db": grad_b,
        }
    )
    if epoch < N_EPOCHS:
        # - atualizar os parametros
        w_modelo = w_modelo - LEARNING_RATE * grad_w
        b_modelo = b_modelo - LEARNING_RATE * grad_b

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: w = {w_modelo:.4f}, b = {b_modelo:.4f}, loss = {loss:.4f}"
        )

# salva os dados de treinamento e avaliar
import json

with open("dados_treinamento.json", "w") as f:
    json.dump(dados, f, indent=4)

# gera as metricas
reta_inicial_vs_final(
    dados, type("Dados", (object,), {"x": x, "y": y})(), TRUE_W, TRUE_B
)
evolucao_parametros(dados, TRUE_W, TRUE_B)
create_line_evolution(x, y, dados)
create_loss_evolution(dados)

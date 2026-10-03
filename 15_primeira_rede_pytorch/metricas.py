from pathlib import Path
from typing import List
import json

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

SEED = 40  # 40 35
np.random.seed(SEED)


def evolucao_metricas(train_log: List[dict], modelo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=False)
    losses = [log["loss_medio"] for log in train_log]
    accs = [log["val_acc"] * 100 for log in train_log]
    axes[1].plot(accs, color="tab:blue", linewidth=3)
    axes[1].set_title("Evolução da Acurácia (%)")
    axes[0].plot(losses, color="tab:orange", linewidth=3)
    axes[0].set_title("Evolução da Loss")

    # Removido código relacionado à reta inicial vs final, pois agora estamos apenas plotando métricas de treino e teste.

    for ax in axes:
        ax.set_xlabel("Época")
        ax.set_ylabel("Valor")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{modelo}/metricas/evolucao_metricas.png")
    plt.close(fig)


def zoom_test(train_log: List[dict], modelo):
    accs = [log["val_acc"] * 100 for log in train_log]
    epocas = list(range(len(accs)))

    # pula epoca zero
    accs = accs[1:]
    epocas = epocas[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(epocas, accs, color="tab:blue", linewidth=3)
    plt.title("Evolução da Acurácia (Zoom Test)")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{modelo}/metricas/zoom_test.png")
    plt.close()


def grade_previsoes(
    nome_modelo: str,
    previsoes_teste: List[dict],
    metadados_teste: pd.DataFrame,
    epoca: int = None,
):
    columns = 4
    lines = (len(previsoes_teste) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(8, 2 * lines), squeeze=False)
    flat_axes = axes.ravel()
    for ax, exemplo in zip(flat_axes, previsoes_teste):
        ax.imshow(
            plt.imread(metadados_teste.loc[exemplo["index"], "path"]).squeeze(),
            cmap="gray",
        )
        ax.set_title(
            f"real {exemplo['target']} | pred {exemplo['pred']}\n{exemplo['confianca']:.0%}",
            color="green" if exemplo["correct"] else "red",
            fontsize=9,
        )
        ax.axis("off")
    for ax in flat_axes[len(previsoes_teste) :]:
        ax.axis("off")
    fig.tight_layout()
    if epoca is not None:
        fig.savefig(
            f"{nome_modelo}/metricas/grade_previsoes_epoca_{epoca}.png", dpi=160
        )
    else:
        fig.savefig(f"{nome_modelo}/metricas/grade_previsoes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    MODEL_NAME = "mlp"
    metrics_folder = Path(f"modelos_treinados/{MODEL_NAME}/metricas")
    metrics_folder.mkdir(parents=True, exist_ok=True)

    # carrega os logs
    with open(f"modelos_treinados/{MODEL_NAME}/historico.json", "r") as f:
        logs = json.load(f)

    evolucao_metricas(logs, modelo="modelos_treinados/" + MODEL_NAME)
    zoom_test(logs, modelo="modelos_treinados/" + MODEL_NAME)

    # para analisar as imagens classificadas mapeia as previsoes de teste
    historico_previsoes_path = f"modelos_treinados/{MODEL_NAME}/historico_teste.json"
    with open(historico_previsoes_path, "r") as f:
        historico_teste = json.load(f)
    print(f"Tamanho do historico de teste: {len(historico_teste[0]['dados'])}")

    # precisa dos metadados para montar o relatorio de imagens
    metadados_teste = pd.read_csv("data/metadados_teste.csv")
    # vamos deixar os indices ja escolhidos para acompanhar a performance ao longo das epocas
    indices_escolhidos = np.random.choice(
        list(range(len(historico_teste[0]["dados"]))), size=20, replace=False
    )
    indices_escolhidos = [
        2121,
        5114,
        5463,
        1862,
        5749,
        3629,
        3169,
        2630,
        7113,
        3936,
        9767,
        8669,
        2326,
        3247,
        8229,
        2306,
        5669,
        6933,
        7525,
        2593,
    ]
    # tambem vamos criar um txt com todos os classification_reports
    classification_reports = []
    for idx, dados_teste in enumerate(historico_teste):
        epoca = dados_teste["epoca"]
        previsoes_teste = dados_teste["dados"]
        grade_previsoes(
            nome_modelo="modelos_treinados/" + MODEL_NAME,
            previsoes_teste=[
                previsoes_teste[int(indice)] for indice in indices_escolhidos
            ],
            metadados_teste=metadados_teste,
            epoca=epoca,
        )

        # adiciona o classification_report ao txt
        y_true = [ex["target"] for ex in previsoes_teste]
        y_pred = [ex["pred"] for ex in previsoes_teste]
        report = classification_report(y_true, y_pred)  # , output_dict=True)
        classification_reports.append(f"epoca: {epoca}\n{report}")

    with open(
        f"modelos_treinados/{MODEL_NAME}/metricas/classification_reports.txt", "w"
    ) as f:
        f.writelines("\n\n".join(classification_reports))

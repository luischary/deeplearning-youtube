import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import MNISTDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

LEARNING_RATE = 0.05
NUM_EPOCAS = 5
SEED = 42
MODEL_NAME = "mlp"

model_folder = Path("./modelos_treinados/" + MODEL_NAME)
model_folder.mkdir(parents=True, exist_ok=True)

# seta seed aleatoria para reproducibilidade
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# 3. definir o modelo
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.camadas = nn.Sequential(
            nn.Flatten(),  # [BATCH, n1, 2, 3, 4...] -> [BATCH, n1*2*3*4...]
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.camadas(x)


def evaluate(model, data_loader):
    model.eval()
    soma_loss = 0.0
    acertos = 0
    dados = []
    with torch.no_grad():
        total = 0
        for batch_imagens, batch_labels in tqdm(
            data_loader, desc="Avaliação", total=len(data_loader)
        ):
            batch_imagens = batch_imagens.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            logits = model(batch_imagens)
            loss = F.cross_entropy(logits, batch_labels)
            soma_loss += loss.item()
            acertos += (logits.argmax(dim=1) == batch_labels).sum().item()

            probs = F.softmax(logits, dim=1)  # [BATCH, NUM_CLASSES]
            for idx in range(batch_labels.shape[0]):
                dados.append(
                    {
                        "index": idx + total,
                        "target": batch_labels[idx].cpu().item(),
                        "pred": logits[idx].argmax(dim=0).cpu().item(),
                        "confianca": probs[idx][logits[idx].argmax(dim=0)].cpu().item(),
                        "correct": (logits[idx].argmax(dim=0) == batch_labels[idx])
                        .cpu()
                        .item(),
                    }
                )
            total += batch_labels.shape[0]

    loss_medio = soma_loss / len(data_loader)
    acc = acertos / len(data_loader.dataset)
    return loss_medio, acc, dados


if __name__ == "__main__":
    # 1. preparar e inspecionar os dados

    # 2. criar Dataset e DataLoader
    df_train = pd.read_csv("./data/metadados_treino.csv")
    train_dataset = MNISTDataset(df_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    df_test = pd.read_csv("./data/metadados_teste.csv")
    test_dataset = MNISTDataset(df_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model = Modelo()
    model = model.to(DEVICE)
    # 4. escolher loss e optimizer
    # loss = F.cross_entropy(logits, labels)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    contagem = 0
    historico = []
    historico_teste = []
    # testa o modelo antes do treinamento
    loss_medio, acc, dados_teste = evaluate(model, test_loader)
    print(f"Antes do treinamento - Val Loss: {loss_medio:.4f}, Val Acc: {acc:.4f}")
    historico.append(
        {"epoca": 0, "loss_medio": None, "val_loss": loss_medio, "val_acc": acc}
    )
    historico_teste.append({"epoca": 0, "dados": dados_teste})

    # 5. treinar por batches e épocas
    for epoca in range(NUM_EPOCAS):
        model.train()
        soma_loss = 0.0
        for batch_imagens, batch_labels in tqdm(
            train_loader, desc=f"Época {epoca+1}/{NUM_EPOCAS}", total=len(train_loader)
        ):
            batch_imagens = batch_imagens.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_imagens)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            contagem += 1
            soma_loss += loss.item()

        loss_medio, acc, dados_teste = evaluate(model, test_loader)
        print(
            f"Época [{epoca+1}/{NUM_EPOCAS}] Loss médio: {soma_loss/len(train_loader):.4f}, "
            f"Val Loss: {loss_medio:.4f}, Val Acc: {acc:.4f}"
        )
        historico.append(
            {
                "epoca": epoca + 1,
                "loss_medio": soma_loss / len(train_loader),
                "val_loss": loss_medio,
                "val_acc": acc,
            }
        )
        historico_teste.append({"epoca": epoca + 1, "dados": dados_teste})

    from pprint import pprint

    pprint(historico)

    # salva os históricos de treinamento e teste em json
    with open(model_folder / "historico.json", "w") as f:
        json.dump(historico, f)

    with open(model_folder / "historico_teste.json", "w") as f:
        json.dump(historico_teste, f)

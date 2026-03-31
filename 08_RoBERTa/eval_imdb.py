import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from src.dataset import DatasetMLM, DatasetClassificacao, DatasetMLMTokenized
from src.model import EncoderMLM, EncoderForClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    MAX_LEN = 512

    base = pd.read_parquet("data/imdb/base_imdb.pq").reset_index(drop=True)
    teste = base[base.split == "test"].reset_index(drop=True)

    test_dataloader = DataLoader(
        dataset=DatasetClassificacao(
            teste,
            max_len=MAX_LEN,
            vocab_size=5_000,
            tokenizer_merges_path="artifacts/tokenizer_wiki_en_5k.json",
        ),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    modelo = EncoderForClassification(
        vocab_size=5_000,
        embed_dim=256,
        num_heads=8,
        hidden_size=1024,
        num_layers=4,
        max_len=MAX_LEN,
        dropout=0.1,
        num_classes=1,
    )

    dados = torch.load(
        "modelos_treinados/imdb_classifier/modelo_treinado.pt",
        map_location="cpu",
    )
    modelo.load_state_dict(dados)
    modelo.to(device)
    modelo.eval()

    labels = []
    probas = []
    preds = []
    with torch.no_grad():
        for dados in tqdm(test_dataloader):
            x, y = dados
            x = x.to(device)
            y = y
            saida = modelo(x)
            probas_batch = torch.sigmoid(saida).squeeze().cpu().numpy()
            preds_batch = (probas_batch > 0.5).astype(int)
            labels.extend(y.cpu().numpy())
            probas.extend(probas_batch)
            preds.extend(preds_batch)

    # encontra o melhor corte para maximizar a f1-score
    from sklearn.metrics import f1_score

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds_threshold = (np.array(probas) > threshold).astype(int)
        f1 = f1_score(labels, preds_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best F1: {best_f1:.4f} at threshold: {best_threshold:.2f}")
    print(
        classification_report(labels, (np.array(probas) > best_threshold).astype(int))
    )

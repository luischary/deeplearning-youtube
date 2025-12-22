import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class DatasetGeracao(Dataset):
    def __init__(self, dados: pd.DataFrame, max_len: int):
        self.dados = dados
        self.max_len = max_len
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        texto = self.dados.loc[idx, "texto"]
        tokens = self.tokenizer.tokenize(texto)
        tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]

        # tamanho
        limite = self.max_len + 1
        tamanho = len(tokens)

        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [PAD_TOKEN for _ in range(diff)]

        x = tokens[:-1]
        y = tokens[1:]

        x = torch.tensor(x, dtype=torch.int)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

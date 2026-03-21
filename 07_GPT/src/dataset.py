import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tokenizer import BPETokenizer, SPECIAL_TOKENS


class DatasetGeracao(Dataset):
    def __init__(self, dados: pd.DataFrame, max_len: int, vocab_size: int = 10_000):
        self.dados = dados
        self.max_len = max_len
        self.tokenizer = BPETokenizer(
            merges_path="artifacts/bpe_tokenizer_100k.json", vocab_size=vocab_size
        )

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        texto = self.dados.loc[index, "texto"]
        tokens = self.tokenizer.encode(texto)

        # bos e eos
        tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens + [SPECIAL_TOKENS["<EOS>"]]

        # arruma o tamanho
        tamanho = len(tokens)
        limite = self.max_len + 1
        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"] for _ in range(diff)]

        x = tokens[:-1]
        y = tokens[1:]

        x = torch.tensor(x, dtype=torch.int)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


class DatasetGeracaoTokenized(Dataset):
    def __init__(self, dados: pd.DataFrame, max_len: int):
        self.max_len = max_len
        # quebra as particoes em pedacos utilizaveis
        pedacos = []
        for r in dados.itertuples():
            particao = r.tokens_path

            for i in range(0, r.tokens_len, max_len + 1):
                pedaco = (particao, i, min(i + max_len + 1, r.tokens_len))
                pedacos.append(pedaco)

        self.dados = pedacos

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        particao, start, end = self.dados[index]
        tokens = np.load(particao, mmap_mode="r").tolist()
        tokens = tokens[start:end]

        # arruma o tamanho
        tamanho = len(tokens)
        limite = self.max_len + 1
        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"] for _ in range(diff)]

        x = tokens[:-1]
        y = tokens[1:]

        x = torch.tensor(x, dtype=torch.int)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

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


class DatasetMLM(Dataset):
    def __init__(
        self,
        dados: pd.DataFrame,
        max_len: int,
        vocab_size: int = 10_000,
        mlm_prob: float = 0.15,
    ):
        self.dados = dados
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.vocab_size = vocab_size
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
        limite = self.max_len
        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"] for _ in range(diff)]

        x = torch.tensor(tokens, dtype=torch.long)

        # agora vamos para o mascaramento
        # IMPORTANTE: Não podemos mascarar tokens especiais ([CLS], [SEP], [PAD])
        special_ids = torch.tensor(list(SPECIAL_TOKENS.values()))
        special_tokens_mask = torch.isin(x, special_ids)

        probs = torch.full(x.shape, self.mlm_prob)
        probs.masked_fill_(special_tokens_mask == 1, 0.0)

        # seleciona quais indices serao mascarados
        masked_indices = torch.bernoulli(probs).bool()
        # Apenas os tokens selecionados terão o valor real na label.
        # O resto vira -100 (índice padrão do PyTorch para ignorar no cálculo da perda/loss)
        labels = x.clone()
        labels[~masked_indices] = -100

        # 1. 80% das vezes, substituímos por [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        x[indices_replaced] = SPECIAL_TOKENS["<MASK>"]

        # 2. 10% das vezes, substituímos por um token aleatório do vocabulário
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        x[indices_random] = random_words[indices_random]

        # 3. Os 10% restantes (implícitos) mantêm o token original
        # Não precisamos fazer nada, pois o token já está lá no 'x' original
        return x, labels


class DatasetMLMTokenized(Dataset):
    def __init__(
        self,
        dados: pd.DataFrame,
        max_len: int,
        vocab_size: int = 10_000,
        mlm_prob: float = 0.15,
    ):
        self.dados = dados
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.vocab_size = vocab_size

        pedacos = []
        for r in dados.itertuples():
            particao = r.tokens_path

            for i in range(0, r.tokens_len, max_len):
                pedaco = (particao, i, min(i + max_len, r.tokens_len))
                pedacos.append(pedaco)

        self.dados = pedacos

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        particao, start, end = self.dados[index]
        tokens = np.load(particao, mmap_mode="r").tolist()[start:end]

        # arruma o tamanho
        tamanho = len(tokens)
        limite = self.max_len
        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"] for _ in range(diff)]

        x = torch.tensor(tokens, dtype=torch.long)

        # agora vamos para o mascaramento
        # IMPORTANTE: Não podemos mascarar tokens especiais ([CLS], [SEP], [PAD])
        special_ids = torch.tensor(list(SPECIAL_TOKENS.values()))
        special_tokens_mask = torch.isin(x, special_ids)

        probs = torch.full(x.shape, self.mlm_prob)
        probs.masked_fill_(special_tokens_mask == 1, 0.0)

        # seleciona quais indices serao mascarados
        masked_indices = torch.bernoulli(probs).bool()
        # Apenas os tokens selecionados terão o valor real na label.
        # O resto vira -100 (índice padrão do PyTorch para ignorar no cálculo da perda/loss)
        labels = x.clone()
        labels[~masked_indices] = -100

        # 1. 80% das vezes, substituímos por [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        x[indices_replaced] = SPECIAL_TOKENS["<MASK>"]

        # 2. 10% das vezes, substituímos por um token aleatório do vocabulário
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        x[indices_random] = random_words[indices_random]

        # 3. Os 10% restantes (implícitos) mantêm o token original
        # Não precisamos fazer nada, pois o token já está lá no 'x' original
        return x, labels


class DatasetClassificacao(Dataset):
    def __init__(
        self,
        dados: pd.DataFrame,
        max_len: int,
        vocab_size: int = 10_000,
        tokenizer_merges_path="artifacts/bpe_tokenizer_100k.json",
    ):
        self.dados = dados
        self.max_len = max_len
        self.tokenizer = BPETokenizer(
            merges_path=tokenizer_merges_path, vocab_size=vocab_size
        )

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        texto = self.dados.loc[index, "texto"]
        label = self.dados.loc[index, "label"]
        if "tokens" in self.dados.columns:
            tokens = list(self.dados.loc[index, "tokens"])
        else:
            tokens = self.tokenizer.encode(texto)

        # bos e eos
        tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens + [SPECIAL_TOKENS["<EOS>"]]

        # arruma o tamanho
        tamanho = len(tokens)
        limite = self.max_len
        if tamanho > limite:
            tokens = tokens[:limite]
        elif tamanho < limite:
            diff = limite - tamanho
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"] for _ in range(diff)]

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


import numpy as np


class DatasetDiffusion(Dataset):
    def __init__(self, dados, max_len, vocab_size=10_000):
        self.dados = dados
        self.max_len = max_len
        self.tokenizer = BPETokenizer(
            merges_path="artifacts/bpe_tokenizer_100k.json", vocab_size=vocab_size
        )
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        texto = self.dados.loc[index, "texto"]
        tokens = self.tokenizer.encode(texto)

        # Adiciona BOS e EOS
        tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens + [SPECIAL_TOKENS["<EOS>"]]

        # Ajuste de tamanho
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        else:
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"]] * (self.max_len - len(tokens))

        x_0 = torch.tensor(tokens, dtype=torch.long)

        # 1. Amostragem de t ~ U(0, 1) conforme o paper [cite: 103, 149]
        t = np.random.rand()

        # 2. Identifica tokens especiais para ignorar [cite: 118, 121]
        special_ids = torch.tensor([SPECIAL_TOKENS["<BOS>"]])
        special_tokens_mask = torch.isin(x_0, special_ids)

        # 3. Lógica de Mascaramento LLaDA:
        # Cada token é mascarado independentemente com probabilidade 't'
        probs = torch.full(x_0.shape, t)
        probs.masked_fill_(special_tokens_mask, 0.0)  # Nunca mascara tokens especiais

        masked_indices = torch.bernoulli(probs).bool()

        # x_t é a nossa entrada (sequência ruidosa)
        x_t = x_0.clone()
        x_t[masked_indices] = SPECIAL_TOKENS["<MASK>"]

        # Labels: No LLaDA, a loss é calculada APENAS nos tokens mascarados [cite: 117, 121]
        labels = x_0.clone()
        # labels[~masked_indices] = (
        #     -100
        # )  # Ignora o que não foi mascarado no cálculo da Loss

        return x_t, labels, torch.tensor(t, dtype=torch.float32)


class DatasetDiffusionTokenized(Dataset):
    def __init__(self, dados, max_len, vocab_size=10_000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        # quebra as particoes em pedacos utilizaveis
        pedacos = []
        for r in dados.itertuples():
            particao = r.tokens_path

            for i in range(0, r.tokens_len, max_len):
                pedaco = (particao, i, min(i + max_len, r.tokens_len))
                pedacos.append(pedaco)

        self.dados = pedacos
        self.current_partition = None
        self.partition = None

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, index):
        particao, start, end = self.dados[index]
        if particao == self.current_partition:
            tokens = self.partition
        else:
            tokens = np.load(particao, mmap_mode="r").tolist()
            self.partition = tokens
            self.current_partition = particao

        tokens = tokens[start:end]

        # Ajuste de tamanho
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        else:
            tokens = tokens + [SPECIAL_TOKENS["<PAD>"]] * (self.max_len - len(tokens))

        x_0 = torch.tensor(tokens, dtype=torch.long)

        # 1. Amostragem de t ~ U(0, 1) conforme o paper [cite: 103, 149]
        t = np.random.rand()

        # 2. Identifica tokens especiais para ignorar [cite: 118, 121]
        special_ids = torch.tensor([SPECIAL_TOKENS["<BOS>"], SPECIAL_TOKENS["<PAD>"]])
        special_tokens_mask = torch.isin(x_0, special_ids)

        # 3. Lógica de Mascaramento LLaDA:
        # Cada token é mascarado independentemente com probabilidade 't'
        probs = torch.full(x_0.shape, t)
        probs.masked_fill_(special_tokens_mask, 0.0)  # Nunca mascara tokens especiais

        masked_indices = torch.bernoulli(probs).bool()

        # x_t é a nossa entrada (sequência ruidosa)
        x_t = x_0.clone()
        x_t[masked_indices] = SPECIAL_TOKENS["<MASK>"]

        # Labels: No LLaDA, a loss é calculada APENAS nos tokens mascarados [cite: 117, 121]
        labels = x_0.clone()
        labels[~masked_indices] = (
            -100
        )  # Ignora o que não foi mascarado no cálculo da Loss

        return x_t, labels, torch.tensor(t, dtype=torch.float32)

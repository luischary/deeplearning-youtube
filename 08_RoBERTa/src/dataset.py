import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tokenizer import BPETokenizer, SPECIAL_TOKENS


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

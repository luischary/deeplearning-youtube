import random

import torch
import pandas as pd
from torch.utils.data import Dataset
from src.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from src.augmentation import text_augmentation


class DatasetCorretor(Dataset):
    def __init__(
        self,
        dados: pd.DataFrame,
        max_len: int = 128,
        augment: bool = False,
        mask: bool = False,
    ):
        self.dados = dados
        self.tokenizer = Tokenizer()
        self.max_len = max_len
        self.augment = augment
        self.mask = mask

    def __len__(self):
        return len(self.dados)

    def arruma_tamanho(self, tokens, max_len):
        tamanho = len(tokens)
        if tamanho < max_len:
            tokens += [PAD_TOKEN] * (max_len - tamanho)
        else:
            tokens = tokens[:max_len]
        return tokens

    def mask_tokens(self, tokens, unk_token, proba):
        return [unk_token if random.random() < proba else token for token in tokens]

    def __getitem__(self, idx):
        texto = self.dados.loc[idx, "text"]
        if self.augment:
            augmented = text_augmentation(texto)
        else:
            augmented = texto

        # X encoder
        if self.mask:
            tokens_aug = self.mask_tokens(
                self.tokenizer.tokenize(augmented), UNK_TOKEN, 0.1
            )
        else:
            tokens_aug = self.tokenizer.tokenize(augmented)

        tokens_aug = [BOS_TOKEN] + tokens_aug + [EOS_TOKEN]
        tokens_aug = self.arruma_tamanho(tokens_aug, self.max_len)
        tokens_aug_t = torch.tensor(tokens_aug, dtype=torch.int)

        # X e Y do decoder
        tokens_texto = [BOS_TOKEN] + self.tokenizer.tokenize(texto) + [EOS_TOKEN]
        tokens_texto = self.arruma_tamanho(tokens_texto, self.max_len + 1)

        x_decoder = tokens_texto[:-1]
        x_decoder_t = torch.tensor(x_decoder, dtype=torch.int)

        y_decoder = tokens_texto[1:]
        y_decoder_t = torch.tensor(y_decoder, dtype=torch.long)

        return tokens_aug_t, x_decoder_t, y_decoder_t


class DatasetValidaCorretor(Dataset):
    def __init__(
        self,
        dados: pd.DataFrame,
        max_len: int = 128,
    ):
        self.dados = dados
        self.tokenizer = Tokenizer()
        self.max_len = max_len

    def __len__(self):
        return len(self.dados)

    def arruma_tamanho(self, tokens, max_len):
        tamanho = len(tokens)
        if tamanho < max_len:
            tokens += [PAD_TOKEN] * (max_len - tamanho)
        else:
            tokens = tokens[:max_len]
        return tokens

    def __getitem__(self, idx):
        texto = self.dados.loc[idx, "text"]
        augmented = self.dados.loc[idx, "augmented_text"]

        # X encoder
        tokens_aug = self.tokenizer.tokenize(augmented)

        tokens_aug = [BOS_TOKEN] + tokens_aug + [EOS_TOKEN]
        tokens_aug = self.arruma_tamanho(tokens_aug, self.max_len)
        tokens_aug_t = torch.tensor(tokens_aug, dtype=torch.int)

        # X e Y do decoder
        tokens_texto = [BOS_TOKEN] + self.tokenizer.tokenize(texto) + [EOS_TOKEN]
        tokens_texto = self.arruma_tamanho(tokens_texto, self.max_len + 1)

        x_decoder = tokens_texto[:-1]
        x_decoder_t = torch.tensor(x_decoder, dtype=torch.int)

        y_decoder = tokens_texto[1:]
        y_decoder_t = torch.tensor(y_decoder, dtype=torch.long)

        return tokens_aug_t, x_decoder_t, y_decoder_t

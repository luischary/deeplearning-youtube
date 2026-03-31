from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import DatasetMLM, DatasetClassificacao, DatasetMLMTokenized
from src.model import EncoderMLM, EncoderForClassification
from src.scheduler import LinearWarmupScheduler
from src.tokenizer import BPETokenizer, SPECIAL_TOKENS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
MODELS_FOLDER = "./modelos_treinados"


@torch.no_grad()
def valid(model, valid_dataloader):
    model.eval()
    losses = []
    for batch in valid_dataloader:
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)

        loss = model.valid_step(batch)
        mask = loss != 0.0
        losses += loss[mask].detach().cpu().tolist()

    return np.mean(losses)


def train(
    model_name,
    model,
    optimizer,
    shcheduler,
    max_steps,
    train_dataloader,
    valid_dataloader,
    early_stop_patience: int = 3,
    eval_steps: int = 1000,
):
    model_folder = Path(MODELS_FOLDER) / model_name
    model_folder.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(model_folder.as_posix())

    steps = 0
    best_valid = None
    valid_count = 0
    model.train()
    optimizer.zero_grad()
    for epoca in range(9999999999):
        for batch in tqdm(train_dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)

            loss = model.train_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            shcheduler.step()
            optimizer.zero_grad()

            steps += 1

            if steps % 25 == 0:
                writer.add_scalar(
                    "train/loss", loss.detach().cpu().item(), global_step=steps
                )
                writer.add_scalar(
                    "learning_rate", shcheduler.get_last_lr()[0], global_step=steps
                )

            if steps % eval_steps == 0:
                valid_loss = valid(model, valid_dataloader)
                writer.add_scalar("valid/loss", valid_loss, global_step=steps)

                torch.save(
                    model.state_dict(),
                    model_folder.as_posix() + "/last.pt",
                )

                if best_valid is None:
                    best_valid = valid_loss
                elif valid_loss < best_valid:
                    torch.save(
                        model.state_dict(),
                        model_folder.as_posix() + "/modelo_treinado.pt",
                    )
                    valid_count = 0
                else:
                    valid_count += 1

                model.train()

            if valid_count >= early_stop_patience or steps >= max_steps:
                print("EARLY STOP!!!")
                break
        if valid_count >= early_stop_patience or steps >= max_steps:
            break


if __name__ == "__main__":
    MAX_LEN = 512

    base = pd.read_parquet("data/imdb/base_imdb.pq").reset_index(drop=True)
    print(base)
    treino = base[base.split == "train"].reset_index(drop=True)
    validacao = base[base.split == "valid"].reset_index(drop=True)

    # treino = pd.read_parquet("data/wiki_en/sample_treino_tokenized.pq").reset_index(
    #     drop=True
    # )
    # validacao = pd.read_parquet(
    #     "data/wiki_en/sample_validacao_tokenized.pq"
    # ).reset_index(drop=True)

    train_dataloader = DataLoader(
        # dataset=DatasetMLM(treino, max_len=MAX_LEN),
        dataset=DatasetClassificacao(
            treino,
            max_len=MAX_LEN,
            vocab_size=5_000,
            tokenizer_merges_path="artifacts/tokenizer_wiki_en_5k.json",
        ),
        # dataset=DatasetMLMTokenized(
        #     treino, vocab_size=5_000, max_len=MAX_LEN, mlm_prob=0.15
        # ),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        # dataset=DatasetMLM(validacao, MAX_LEN),
        dataset=DatasetClassificacao(
            validacao,
            max_len=MAX_LEN,
            vocab_size=5_000,
            tokenizer_merges_path="artifacts/tokenizer_wiki_en_5k.json",
        ),
        # DatasetMLMTokenized(
        #     validacao, vocab_size=5_000, max_len=MAX_LEN, mlm_prob=0.15
        # ),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # modelo = EncoderMLM(
    #     vocab_size=5_000,
    #     embed_dim=256,
    #     num_heads=8,
    #     hidden_size=1024,
    #     num_layers=4,
    #     max_len=MAX_LEN,
    #     dropout=0.1,
    # )
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

    optimizer = torch.optim.Adam(modelo.parameters(), lr=3e-5)
    shcheduler = LinearWarmupScheduler(
        optimizer=optimizer, warmup=1000, max_iters=30_000, min_percent=0.001
    )
    modelo.to(device)

    train(
        model_name="imdb_classifier_pretrained_200k",
        model=modelo,
        optimizer=optimizer,
        shcheduler=shcheduler,
        max_steps=30_000,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        early_stop_patience=5,
        eval_steps=500,
    )

    dados = torch.load(
        "modelos_treinados/encoder_mlm_wiki_en_5k_200k/last.pt",
        map_location="cpu",
    )
    modelo.load_state_dict(dados, strict=False)

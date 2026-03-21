from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import DatasetGeracao, DatasetGeracaoTokenized
from src.model import DecoderLM
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
        mask = (batch[-1].reshape((-1)) != SPECIAL_TOKENS["<PAD>"]) & (loss != 0.0)
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
    accum_steps: int = 1,
):
    model_folder = Path(MODELS_FOLDER) / model_name
    model_folder.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(model_folder.as_posix())

    training_steps = 0
    update_steps = 0
    best_valid = None
    valid_count = 0
    model.train()
    optimizer.zero_grad()
    for epoca in range(9999999999):
        for batch in tqdm(train_dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)

            loss = model.train_step(batch)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            training_steps += 1
            if training_steps % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                shcheduler.step()
                optimizer.zero_grad()

                update_steps += 1

                if update_steps % 25 == 0:
                    writer.add_scalar(
                        "train/loss",
                        loss.detach().cpu().item(),
                        global_step=update_steps,
                    )
                    writer.add_scalar(
                        "learning_rate",
                        shcheduler.get_last_lr()[0],
                        global_step=update_steps,
                    )

                if update_steps % 1000 == 0:
                    valid_loss = valid(model, valid_dataloader)
                    writer.add_scalar(
                        "valid/loss", valid_loss, global_step=update_steps
                    )

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

            if valid_count >= early_stop_patience or update_steps >= max_steps:
                print("EARLY STOP!!!")
                break
        if valid_count >= early_stop_patience or update_steps >= max_steps:
            break


if __name__ == "__main__":
    MAX_LEN = 512

    base = pd.read_parquet("data/base_treino_tokenized.pq")
    print(base)

    treino = base
    validacao = pd.read_parquet("data/base_validacao_tokenized.pq")

    train_dataloader = DataLoader(
        dataset=DatasetGeracaoTokenized(treino, max_len=MAX_LEN),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        dataset=DatasetGeracaoTokenized(validacao, MAX_LEN),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    modelo = DecoderLM(
        vocab_size=10_000,
        embed_dim=256,
        num_heads=8,
        hidden_size=1024,
        num_layers=4,
        max_len=MAX_LEN,
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(modelo.parameters(), lr=3e-4)
    shcheduler = LinearWarmupScheduler(
        optimizer=optimizer, warmup=2000, max_iters=300_000, min_percent=0.001
    )
    modelo.to(device)

    train(
        model_name="teste_decoder_wiki",
        model=modelo,
        optimizer=optimizer,
        shcheduler=shcheduler,
        max_steps=300_000,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        early_stop_patience=50,
        accum_steps=2,
    )

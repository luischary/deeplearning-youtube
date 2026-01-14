from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import DatasetCorretor, DatasetValidaCorretor
from src.model import Corretor, AttentionCorrector
from src.scheduler import LinearWarmupScheduler
from evaluate import wer_metric, cer_metric
from src.tokenizer import Tokenizer, PAD_TOKEN
from inference import generate, generate_with_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer()
MODELS_FOLDER = "./modelos_treinados"


@torch.no_grad()
def valid(model, valid_dataloader):
    model.eval()
    losses = []
    for batch in valid_dataloader:
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)

        loss = model.valid_step(batch)
        mask = batch[-1].reshape((-1)) != PAD_TOKEN
        losses += loss[mask].detach().cpu().tolist()

    return np.mean(losses)


@torch.no_grad()
def valid_cer_wer(model, valid_dataframe):
    model.eval()
    cer_valid = []
    wer_valid = []
    for i in tqdm(range(min(len(valid_dataframe), 500))):
        # limita para nao demorar muito
        augmented_ref = valid_dataframe.loc[i, "augmented_text"]
        target = valid_dataframe.loc[i, "text"]

        generated = generate_with_attention(
            model, augmented_ref, tokenizer, max_len=100
        )
        cer_model = cer_metric(generated, target)
        cer_valid.append(cer_model)

        wer_model = wer_metric(generated, target)
        wer_valid.append(wer_model)
    return np.mean(cer_valid), np.mean(wer_valid)


def train(
    model_name,
    model,
    optimizer,
    shcheduler,
    max_steps,
    train_dataloader,
    valid_dataloader,
):
    model_folder = Path(MODELS_FOLDER) / model_name
    model_folder.mkdir(parents=True, exist_ok=False)
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

            if steps % 1000 == 0:
                valid_loss = valid(model, valid_dataloader)
                writer.add_scalar("valid/loss", valid_loss, global_step=steps)
                valid_cer, valid_wer = valid_cer_wer(
                    model, valid_dataloader.dataset.dados
                )
                writer.add_scalar("valid/CER", valid_cer, global_step=steps)
                writer.add_scalar("valid/WER", valid_wer, global_step=steps)
                combined = valid_cer + valid_wer
                writer.add_scalar("valid/combined", combined, global_step=steps)

                torch.save(
                    model.state_dict(),
                    model_folder.as_posix() + "/last.pt",
                )

                if best_valid is None:
                    best_valid = combined
                elif combined < best_valid:
                    torch.save(
                        model.state_dict(),
                        model_folder.as_posix() + "/modelo_treinado.pt",
                    )
                    valid_count = 0
                else:
                    valid_count += 1

                model.train()

            if valid_count >= 3 or steps >= max_steps:
                print("EARLY STOP!!!")
                break
        if valid_count >= 3 or steps >= max_steps:
            break


if __name__ == "__main__":
    MAX_LEN = 100

    # treino = (
    #     pd.read_parquet("data/train_set.pq").sample(32 * 100_000).reset_index(drop=True)
    # )
    # treino.to_parquet("data/train_set_sample.pq")
    treino = pd.read_parquet("data/train_set_sample.pq")
    validacao = pd.read_parquet("data/validation_set.pq").reset_index(drop=True)

    train_dataloader = DataLoader(
        dataset=DatasetCorretor(treino, MAX_LEN, augment=True, mask=True),
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    valid_dataloader = DataLoader(
        dataset=DatasetValidaCorretor(validacao, MAX_LEN),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # modelo = Corretor(
    #     vocab_size=100 + 4,
    #     embed_dim=256,
    #     hidden_encoder=512,
    #     hidden_decoder=512,
    #     num_layers_encoder=2,
    #     num_layers_decoder=2,
    # )
    modelo = AttentionCorrector(
        vocab_size=100 + 4,
        embed_dim=128,
        hidden_encoder=256,
        hidden_decoder=256,
        num_layers_encoder=1,
        num_layers_decoder=1,
    )
    optimizer = torch.optim.Adam(modelo.parameters(), lr=7e-4)
    shcheduler = LinearWarmupScheduler(
        optimizer=optimizer, warmup=2_500, max_iters=100_000, min_percent=0.001
    )
    modelo.to(device)

    train(
        model_name="grande_256_512_2l",
        model=modelo,
        optimizer=optimizer,
        shcheduler=shcheduler,
        max_steps=100_000,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )

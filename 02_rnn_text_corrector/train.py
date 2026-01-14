import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DatasetCorretor, DatasetValidaCorretor
from src.model import Corretor, AttentionCorrector

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def valid(model, valid_dataloader):
    model.eval()
    losses = []
    for batch in valid_dataloader:
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)

        loss = model.valid_step(batch)
        losses += loss.detach().cpu().tolist()

    return np.mean(losses)


def train(model, optimizer, num_epochs, train_dataloader, valid_dataloader):
    steps = 0
    best_valid = None
    valid_count = 0
    model.train()
    optimizer.zero_grad()
    for epoca in range(num_epochs):
        for batch in train_dataloader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)

            loss = model.train_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            steps += 1

            if steps % 100 == 0:
                print(
                    f"Epoca: {epoca} - {steps}\tTrain_loss: {loss.detach().cpu().item():.4f}"
                )

            if steps % 1000 == 0:
                valid_loss = valid(model, valid_dataloader)
                print(f"\nEpoca: {epoca} - {steps}\tValid_loss: {valid_loss:.4f}\n")

                if best_valid is None:
                    best_valid = valid_loss
                elif valid_loss < best_valid:
                    torch.save(model.state_dict(), "modelo_treinado_big.pt")
                    valid_count = 0
                else:
                    valid_count += 1

                model.train()

            if valid_count >= 3 or steps >= 100_000:
                print("EARLY STOP!!!")
                break
        if valid_count >= 3 or steps >= 100_000:
            break


if __name__ == "__main__":
    MAX_LEN = 100

    treino = pd.read_parquet("data/train_set.pq").reset_index(drop=True)
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
        embed_dim=256,
        hidden_encoder=512,
        hidden_decoder=512,
        num_layers_encoder=2,
        num_layers_decoder=2,
    )
    optimizer = torch.optim.Adam(modelo.parameters(), lr=7e-4)

    modelo.to(device)

    train(
        model=modelo,
        optimizer=optimizer,
        num_epochs=1,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )

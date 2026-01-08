import pandas as pd

from src.augmentation import text_augmentation

if __name__ == "__main__":
    # Carregando apenas a coluna text para economizar RAM na leitura
    df = pd.read_parquet("data/base_treinamento_nodup.pq", columns=["text"])

    df = df.sample(frac=1.0, replace=False).reset_index(drop=True)
    split = ["valid" for _ in range(1500)] + ["train" for _ in range(len(df) - 1500)]
    df["split"] = split

    valid = df[df["split"] == "valid"].reset_index(drop=True)
    valid["augmented_text"] = valid["text"].apply(text_augmentation)
    valid.to_parquet("data/validation_set.pq")

    df[df.split == "train"].to_parquet("data/train_set.pq")

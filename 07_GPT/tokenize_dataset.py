from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.tokenizer import BPETokenizer, SPECIAL_TOKENS

tokenizer = BPETokenizer(
    merges_path="artifacts/bpe_tokenizer_100k.json", vocab_size=10_000
)


def run_tokenization_on_df(df, split, PARTITION_SIZE, OUTPUT_PATH):
    idx = 0
    current_tokens = []
    metadata = {
        "tokens_path": [],
        "tokens_len": [],
        "split": [],
    }
    save_folder = OUTPUT_PATH / split
    save_folder.mkdir(parents=True, exist_ok=True)

    for r in tqdm(df.itertuples(), total=len(df)):
        texto = r.texto
        tokens = (
            [SPECIAL_TOKENS["<BOS>"]]
            + tokenizer.encode(texto)
            + [SPECIAL_TOKENS["<EOS>"]]
        )

        current_tokens += tokens
        while len(current_tokens) > PARTITION_SIZE:
            export = current_tokens[:PARTITION_SIZE]
            tokens_np = np.asarray(export, dtype=np.int32)

            path_tokens = OUTPUT_PATH / split / f"{idx}.npy"
            np.save(path_tokens.as_posix(), tokens_np, allow_pickle=False)

            metadata["tokens_path"].append(path_tokens.as_posix())
            metadata["tokens_len"].append(len(export))
            metadata["split"].append(split)

            idx += 1
            current_tokens = current_tokens[PARTITION_SIZE:]

    # ultima particao
    if len(current_tokens) > 0:
        tokens_np = np.asarray(current_tokens, dtype=np.int32)

        path_tokens = OUTPUT_PATH / split / f"{idx}.npy"
        np.save(path_tokens.as_posix(), tokens_np, allow_pickle=False)

        metadata["tokens_path"].append(path_tokens.as_posix())
        metadata["tokens_len"].append(len(current_tokens))
        metadata["split"].append(split)

    return pd.DataFrame(metadata)


def run_tokenization_parallel(
    df, split, PARTITION_SIZE, OUTPUT_PATH, batch_size, num_workers=10
):
    batches = []
    for i in range(0, len(df), batch_size):
        batches.append(df.iloc[i : i + batch_size])

    # payloads = []
    splits = [f"{split}_{idx}" for idx in range(len(batches))]
    partitions = [PARTITION_SIZE for _ in range(len(batches))]
    outputs = [OUTPUT_PATH for _ in range(len(batches))]
    # for idx, batch in enumerate(batches):
    # payloads.append((batch, f"{split}_{idx}", PARTITION_SIZE, OUTPUT_PATH))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    run_tokenization_on_df, batches, splits, partitions, outputs
                ),
                total=len(batches),
                desc="Tokenizando dataset",
            )
        )
        return results


if __name__ == "__main__":
    OUTPUT_PATH = Path("artifacts/tokenized_dataset")
    PARTITION_SIZE = 50_000
    df = pd.read_parquet("data/base_treino_wiki_reviews.pq")
    treino = df[df.split == "train"].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)

    metadados_valid = run_tokenization_on_df(
        valid, "valid", PARTITION_SIZE, OUTPUT_PATH
    )
    print(metadados_valid)
    metadados_valid.to_parquet("data/base_validacao_tokenized.pq")

    metadados_train = run_tokenization_parallel(
        treino,
        "train",
        PARTITION_SIZE,
        OUTPUT_PATH,
        batch_size=30_000,
        num_workers=15,
    )
    metadados_train = pd.concat(metadados_train).reset_index(drop=True)
    metadados_train["split"] = ["train" for _ in range(len(metadados_train))]
    print(metadados_train)
    metadados_train.to_parquet("data/base_treino_tokenized.pq")

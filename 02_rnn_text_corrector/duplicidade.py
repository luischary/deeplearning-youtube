import re
from hashlib import md5
from typing import List, Set

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import pandas as pd

# CONFIGURAÇÕES
NUM_PERM = 128  # Padrão da indústria (equilíbrio precisão vs memória)
THRESHOLD = 0.85  # 85% de similaridade = duplicata (ajuste conforme calibração)
SHINGLE_SIZE = 3  # N-grams (analisa triplas de palavras)


def preprocess_text(text: str) -> str:
    # Normalização leve para melhorar o match
    text = text.lower().strip()
    # Remove pontuação básica para focar nas palavras
    text = re.sub(r"[^\w\s]", "", text)
    return text


def get_shingles(text: str, n: int = 3) -> Set[str]:
    shingles = set()
    if len(text) < n:
        shingles.add(text)
        return shingles
    else:
        for i in range(len(text) - n + 1):
            shingle = text[i : i + n]
            shingles.add(shingle)
    return shingles


def compute_minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    shingles = get_shingles(preprocess_text(text), n=SHINGLE_SIZE)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m


def deduplicate_dataset(sentences: List[str]):
    print(f"Dataset original: {len(sentences)} sentenças")

    # 1. DEDUPLICAÇÃO EXATA (FAST TRACK)
    # Remove o grosso ("Obrigado", "Sim", "Não") instantaneamente
    print("Etapa 1: Deduplicação Exata (Hash simples)...")
    seen_exact = set()
    unique_exact = []

    for s in tqdm(sentences, desc="Exact Dedup"):
        s_clean = s.strip()
        if not s_clean:
            continue  # Pula vazios

        # Hash rápido
        hash_s = md5(s_clean.encode("utf8")).hexdigest()
        if hash_s not in seen_exact:
            seen_exact.add(hash_s)
            unique_exact.append(s_clean)

    print(f"Após deduplicação exata: {len(unique_exact)} sentenças únicas")

    # 2. DEDUPLICAÇÃO APROXIMADA (MINHASH + LSH)
    print("Etapa 2: Deduplicação Aproximada (MinHash + LSH)...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}
    unique_approx = []

    for i, s in enumerate(tqdm(unique_exact, desc="Computando MinHash")):
        m = compute_minhash(s, NUM_PERM)
        minhashes[i] = m

    for i, s in enumerate(tqdm(unique_exact, desc="Approx Dedup")):
        m = minhashes[i]
        result = lsh.query(m)
        if not result:
            lsh.insert(i, m)
            unique_approx.append(s)

    print(f"Após deduplicação aproximada: {len(unique_approx)} sentenças únicas")
    return unique_approx


if __name__ == "__main__":
    # Carregando apenas a coluna text para economizar RAM na leitura
    df = pd.read_parquet("data/base_treinamento.pq", columns=["text"]).sample(
        30_000, replace=False
    )

    clean_data = deduplicate_dataset(df["text"].tolist())

    print(f"Dataset Final: {len(clean_data)} sentenças.")

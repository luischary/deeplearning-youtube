import re
from typing import List, Set
import numpy as np
import pandas as pd
from datasketch import MinHash
from tqdm import tqdm
from hashlib import md5
import gc
import os

# --- CONFIGURAÇÕES ---
NUM_PERM = 128
THRESHOLD = 0.85
SHINGLE_SIZE = 3
FILE_PATH = "data/base_treinamento.pq"
MEMMAP_FILENAME = "minhash_sigs.dat"

# Calcula bandas e linhas por banda para atingir o Threshold desejado
# Aproximação LSH: (1/b)^(1/r) ~ threshold
# Para 128 perms e 0.85, b=20 e r=6 costuma funcionar bem, ou b=16 r=8.
# Vamos usar o padrão do datasketch para 0.85
BANDS = 20
ROWS_PER_BAND = NUM_PERM // BANDS


def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
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


def compute_signature(text: str, num_perm: int) -> np.ndarray:
    # Cria o objeto, extrai o array e mata o objeto para economizar RAM
    m = MinHash(num_perm=num_perm)
    for s in get_shingles(preprocess_text(text), n=SHINGLE_SIZE):
        m.update(s.encode("utf8"))
    # Retorna apenas o array numpy (muito leve)
    return m.hashvalues


def deduplicate_at_scale(df: pd.DataFrame):
    n_samples = len(df)
    print(f"Processando {n_samples} textos...")

    # 1. DEDUPLICAÇÃO EXATA (MD5) - Mantive sua lógica, é ótima.
    print(">>> Etapa 1: Exact Dedup (MD5)")
    seen_hashes = set()
    keep_indices = []

    # Vamos iterar apenas índices para economizar
    texts = df["text"].values

    for idx, text in enumerate(tqdm(texts, desc="Exact MD5")):
        if not isinstance(text, str) or not text.strip():
            continue

        h = md5(text.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            keep_indices.append(idx)

    print(f"Reduzido de {n_samples} para {len(keep_indices)} únicos exatos.")

    # Filtra o dataframe para trabalhar apenas com os únicos exatos
    # Reset index é crucial aqui para alinhar com a matriz memmap
    df_unique = df.iloc[keep_indices].reset_index(drop=True)
    del seen_hashes, keep_indices, texts
    gc.collect()

    n_unique = len(df_unique)

    # 2. CALCULAR ASSINATURAS E SALVAR EM DISCO (Memmap)
    # Criamos um arquivo no disco que finge ser um array na RAM.
    # Tamanho: N_linhas x 128 inteiros (uint64 requer 8 bytes, uint32 requer 4)
    # MinHash usa uint64 por padrão.
    print(">>> Etapa 2: Computando Assinaturas MinHash (Disk-backed)")

    if os.path.exists(MEMMAP_FILENAME):
        os.remove(MEMMAP_FILENAME)

    # Aloca espaço em disco
    sigs_mmap = np.memmap(
        MEMMAP_FILENAME, dtype="uint64", mode="w+", shape=(n_unique, NUM_PERM)
    )

    batch_size = 10000
    texts_unique = df_unique["text"].values

    for i in tqdm(range(0, n_unique, batch_size), desc="Hashing"):
        batch_texts = texts_unique[i : i + batch_size]
        batch_sigs = []
        for text in batch_texts:
            # Pula textos muito curtos para evitar falsos positivos agressivos
            if len(text) < 15:
                # Gera uma assinatura aleatória/vazia para não colidir
                batch_sigs.append(
                    np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype="uint64")
                )
            else:
                batch_sigs.append(compute_signature(text, NUM_PERM))

        # Escreve direto no disco
        sigs_mmap[i : i + len(batch_sigs)] = np.array(batch_sigs, dtype="uint64")

    # Flush para garantir escrita
    sigs_mmap.flush()

    # 3. LSH VIA ORDENAÇÃO DE BANDAS (Pure NumPy - No Pandas Overhead)
    print(">>> Etapa 3: Verificando Colisões por Bandas")

    possible_duplicates = set()

    for b in tqdm(range(BANDS), desc="Bands check"):
        start_idx = b * ROWS_PER_BAND
        end_idx = start_idx + ROWS_PER_BAND

        # Carrega a fatia da banda
        band_data = sigs_mmap[:, start_idx:end_idx]

        # 1. View como void (opaco) para comparar a linha inteira de uma vez
        # Isso transforma uma matriz (N, 6) em um vetor (N,) de blobs de bytes
        dtype_void = np.dtype((np.void, band_data.dtype.itemsize * band_data.shape[1]))
        band_void = np.ascontiguousarray(band_data).view(dtype_void).reshape(-1)

        # 2. Encontrar duplicatas via Ordenação (O(N log N))
        # Em vez de hash map (Pandas), ordenamos o array.
        # Valores iguais ficarão adjacentes.

        # Retorna os índices que ordenariam o array
        sort_idx = np.argsort(band_void)

        # Aplica a ordenação no array de valores
        sorted_vals = band_void[sort_idx]

        # 3. Compara vizinhos
        # Cria uma máscara booleana onde True indica que o item é igual ao anterior (duplicata)
        # O primeiro item nunca é duplicata de um anterior, então começamos com False
        dupes_mask = np.empty(len(sorted_vals), dtype=bool)
        dupes_mask[0] = False
        dupes_mask[1:] = sorted_vals[:-1] == sorted_vals[1:]

        # 4. Recupera os índices originais das duplicatas
        # sort_idx[dupes_mask] pega os índices originais dos itens marcados como True
        current_band_dupes = sort_idx[dupes_mask]

        possible_duplicates.update(current_band_dupes)

        # Limpeza agressiva
        del band_data, band_void, sort_idx, sorted_vals, dupes_mask, current_band_dupes
        # gc.collect() # Opcional: descomente se a RAM estiver muito crítica

    print(
        f"Total de Near-Duplicates encontrados (Candidatos): {len(possible_duplicates)}"
    )

    # 4. REMOÇÃO FINAL
    # Mantemos índices que NÃO estão no set de duplicatas
    all_indices = set(range(n_unique))
    keep_indices_final = list(all_indices - possible_duplicates)
    keep_indices_final.sort()

    final_df = df_unique.iloc[keep_indices_final].copy()

    # --- CORREÇÃO DO LOCK NO WINDOWS ---
    # 1. Deletar a referência ao objeto memmap
    del sigs_mmap

    # 2. Forçar o Garbage Collector a rodar AGORA para liberar o handle do arquivo
    gc.collect()

    # 3. Agora o arquivo deve estar destravado para remoção
    try:
        os.remove(MEMMAP_FILENAME)
    except PermissionError:
        print(
            f"Aviso: Não foi possível remover {MEMMAP_FILENAME} automaticamente. Delete manualmente depois."
        )

    return final_df["text"].tolist()


if __name__ == "__main__":
    # Carregando apenas a coluna text para economizar RAM na leitura
    df = pd.read_parquet("data/base_treinamento.pq", columns=["text"])

    clean_data = deduplicate_at_scale(df)

    print(f"Dataset Final: {len(clean_data)} sentenças.")

    # Salvar
    df_final = pd.DataFrame({"text": clean_data})
    df_final.to_parquet("data/base_treinamento_nodup.pq")

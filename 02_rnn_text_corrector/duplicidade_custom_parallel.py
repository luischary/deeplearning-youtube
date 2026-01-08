import re
from typing import List, Set
import numpy as np
import pandas as pd
from datasketch import MinHash
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from hashlib import md5
import gc
import os

# --- CONFIGURAÇÕES ---
NUM_PERM = 128
THRESHOLD = 0.85
SHINGLE_SIZE = 3
FILE_PATH = "base_treinamento.pq"
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


def worker_compute_batch(
    text_list: List[str], num_perm: int, shingle_size: int
) -> np.ndarray:
    """
    Função isolada que roda em cada núcleo da CPU.
    Recebe uma lista de textos crua e retorna uma matriz numpy (N, 128).
    """
    batch_sigs = []
    max_uint64 = np.iinfo(np.uint64).max

    for text in text_list:
        # Lógica de proteção contra textos curtos/vazios
        if len(text) < 15:
            # Assinatura dummy para não quebrar o shape
            batch_sigs.append(np.full(num_perm, max_uint64, dtype="uint64"))
            continue

        # Recriamos a lógica aqui dentro para ser self-contained
        tokens = text.lower().strip().split()  # Simplificado para velocidade
        # Se quiser usar o preprocess_text completo, garanta que ele é importável aqui

        m = MinHash(num_perm=num_perm)
        # Shingling inline para evitar overhead de chamadas de função
        if len(tokens) < shingle_size:
            if tokens:
                m.update(" ".join(tokens).encode("utf8"))
            else:
                m.update(b"_empty_")
        else:
            for i in range(len(tokens) - shingle_size + 1):
                s = " ".join(tokens[i : i + shingle_size])
                m.update(s.encode("utf8"))

        batch_sigs.append(m.hashvalues)

    return np.array(batch_sigs, dtype="uint64")


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
    # 2. CALCULAR ASSINATURAS (PARALELIZADO)
    print(">>> Etapa 2: Computando Assinaturas em Paralelo")

    if os.path.exists(MEMMAP_FILENAME):
        try:
            os.remove(MEMMAP_FILENAME)
        except:
            pass

    sigs_mmap = np.memmap(
        MEMMAP_FILENAME, dtype="uint64", mode="w+", shape=(n_unique, NUM_PERM)
    )

    # Configuração do Paralelismo
    n_jobs = max(1, cpu_count() - 1)  # Deixa 1 core livre para o SO/Spotify
    # Batch size maior para compensar o overhead de criar processos no Windows
    parallel_batch_size = 50000

    texts_unique = df_unique["text"].values
    total_batches = (n_unique + parallel_batch_size - 1) // parallel_batch_size

    # Gerador de chunks para não estourar memória criando listas gigantes
    def batch_generator():
        for i in range(0, n_unique, parallel_batch_size):
            yield texts_unique[i : i + parallel_batch_size]

    # Joblib com generator
    # return_as='generator' é crucial para não guardar todos os resultados na RAM de uma vez!
    # O processamento acontece on-the-fly à medida que iteramos no loop abaixo.
    results_generator = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(worker_compute_batch)(batch, NUM_PERM, SHINGLE_SIZE)
        for batch in batch_generator()
    )

    # Loop de consumo e escrita (Main Thread)
    current_idx = 0
    with tqdm(total=n_unique, desc="Hashing Parallel") as pbar:
        for batch_matrix in results_generator:
            # O worker devolve a matriz pronta (N_batch, 128)
            n_rows = len(batch_matrix)

            # Escrita sequencial no disco (seguro)
            sigs_mmap[current_idx : current_idx + n_rows] = batch_matrix

            # Atualiza índices
            current_idx += n_rows
            pbar.update(n_rows)

            # Flush periódico para garantir que dados vão pro disco
            if current_idx % (parallel_batch_size * 2) == 0:
                sigs_mmap.flush()

    sigs_mmap.flush()
    print("Cálculo de assinaturas concluído.")

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

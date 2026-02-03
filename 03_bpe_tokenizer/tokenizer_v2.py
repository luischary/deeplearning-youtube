import re
import json
from collections import Counter

from tqdm import tqdm

SPLIT_PATTERN = r"""[a-zA-Zà-úÀ-Ú]+\s?|\d|\s|."""


class BPETokenizer:
    def __init__(self, merges_path: str = None):
        self.merges = {}
        self.vocab = {}
        if merges_path:
            self.load_merges(merges_path)

    def load_merges(self, merges_path: str):
        """Carrega os merges e reconstrói o dicionário original."""
        with open(merges_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconvertemos a string "id1,id2" de volta para a tupla (int, int)
        merges = {}
        for key, val in data.items():
            p1, p2 = map(int, key.split(","))
            merges[(p1, p2)] = val

        self.merges = merges
        self.vocab = self.build_vocab(merges)

    def build_vocab(self, merges):
        # O vocabulário inicial são os 256 bytes individuais
        vocab = {i: bytes([i]) for i in range(256)}
        # Adicionamos os merges na ordem em que foram criados
        for (p1, p2), new_id in merges.items():
            vocab[new_id] = vocab[p1] + vocab[p2]
        return vocab

    def save(self, merges_path: str):
        """Salva os merges em um arquivo JSON."""
        with open(merges_path, "w", encoding="utf-8") as f:
            # Convertendo as tuplas de volta para strings "id1,id2"
            serializable_merges = {
                f"{p1},{p2}": v for (p1, p2), v in self.merges.items()
            }
            with open(merges_path, "w", encoding="utf-8") as f:
                json.dump(serializable_merges, f, indent=4)

    def get_stats(self, chunks):
        counts = {}
        for chunk in chunks:
            # Conta pares apenas dentro de cada sub-sequência isolada
            for pair in zip(chunk, chunk[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def get_stats_unique(self, ids_counts):
        """Calcula estatísticas pesadas pela frequência de cada chunk único."""
        counts = {}
        for ids, freq in ids_counts.items():
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    def merge(self, chunks, pair, replacement_id):
        new_chunks = []
        for chunk in chunks:
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                    new_chunk.append(replacement_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_chunks.append(new_chunk)
        return new_chunks

    def merge_ids(self, ids_counts, pair, replacement_id):
        """Aplica o merge apenas nos chunks únicos."""
        new_ids_counts = {}
        for ids, freq in ids_counts.items():
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(replacement_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            # V2, so coloca se o tamanho for maior que 1
            if len(new_ids) > 1:
                new_ids_counts[tuple(new_ids)] = freq
        return new_ids_counts

    def encode(self, text, merges):
        # Começamos com a sequência de bytes brutos
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self.get_stats([tokens])
            # Crucial: encontrar o par com o menor ID de merge (maior prioridade)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))

            # Se o par mais "prioritário" não está nos merges, terminamos
            if pair not in merges:
                break

            # Aplica o merge e repete o processo
            new_id = merges[pair]
            tokens = self.merge([tokens], pair, new_id)

        return tokens

    def decode(self, ids):
        # Concatena os bytes de cada token ID
        tokens_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Transforma de volta em string (UTF-8)
        return tokens_bytes.decode("utf-8", errors="replace")

    def train(self, texts, vocab_size):
        # 1. Pré-tokenização: Contamos a frequência de cada 'palavra' única no dataset
        print("Pré-processando textos...")
        words_counts = Counter()
        for t in texts:
            words_counts.update(re.findall(SPLIT_PATTERN, t))

        # 2. Transformamos as palavras únicas em listas de bytes (tuples)
        # Estrutura: { (byte1, byte2, ...): frequência }
        ids_counts = {
            tuple(w.encode("utf-8")): freq for w, freq in words_counts.items()
        }

        self.merges = {}

        # 3. Loop de Merges
        num_merges = vocab_size - 256
        for i in tqdm(range(num_merges)):
            stats = self.get_stats_unique(ids_counts)
            if not stats:
                break

            top_pair = max(stats, key=stats.get)
            new_id = 256 + i

            # Aplicamos o merge apenas no dicionário de chunks únicos
            ids_counts = self.merge_ids(ids_counts, top_pair, new_id)
            self.merges[top_pair] = new_id

            # if (i + 1) % 10 == 0 or i == num_merges - 1:
            #     print(
            #         f"Merge {i+1}/{num_merges}: {top_pair} -> {new_id} (Frequência: {stats[top_pair]})"
            #     )

        self.vocab = self.build_vocab(self.merges)

    def print_merges(self):
        for (p1, p2), new_id in self.merges.items():
            print(f"({p1}, {p2}) -> {self.decode([new_id])}")

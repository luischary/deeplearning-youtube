import re
import json

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
        chunks = []
        for t in texts:
            chunks.extend(re.findall(SPLIT_PATTERN, t))

        tokens = [
            list(t.encode("utf-8")) for t in chunks
        ]  # Começamos com os bytes (0-255)
        merges = {}  # (p1, p2) -> new_id

        for i in tqdm(range(vocab_size - 256)):
            stats = self.get_stats(tokens)
            top_pair = max(stats, key=stats.get)  # O "caçador de padrões"
            new_id = 256 + i
            tokens = self.merge(tokens, top_pair, new_id)
            merges[top_pair] = new_id
            # print(
            #     f"Merge {i+1}: {top_pair} -> {new_id} (Frequência: {stats[top_pair]})"
            # )

        self.vocab = self.build_vocab(merges)
        self.merges = merges

    def print_merges(self):
        for (p1, p2), new_id in self.merges.items():
            print(f"({p1}, {p2}) -> {self.decode([new_id])}")

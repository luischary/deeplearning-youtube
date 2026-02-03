import re
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm

SPLIT_PATTERN = r"""[a-zA-Zà-úÀ-Ú]+\s?|\d|\s|."""


# Helper function fora da classe para facilitar a serialização (pickling)
def _encode_single_text(text, merges):
    """Lógica de codificação para um único texto."""
    tokens = list(text.encode("utf-8"))

    while len(tokens) >= 2:
        # Encontra todos os pares possíveis na sequência atual
        pairs = set(zip(tokens, tokens[1:]))

        # Filtra apenas os pares que existem nos merges e busca o de menor ID (maior prioridade)
        best_pair = None
        min_id = float("inf")

        for p in pairs:
            if p in merges and merges[p] < min_id:
                best_pair = p
                min_id = merges[p]

        if best_pair is None:
            break

        # Aplica o merge na sequência
        new_id = merges[best_pair]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


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

    def merge(self, chunk, pair, replacement_id):
        new_chunk = []
        i = 0
        while i < len(chunk):
            if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                new_chunk.append(replacement_id)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        return new_chunk

    def merge_ids(self, ids_counts, to_merge):
        """Aplica o merge apenas nos chunks únicos."""
        new_ids_counts = {}
        for ids, freq in ids_counts.items():
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) in to_merge:
                    new_ids.append(to_merge[(ids[i], ids[i + 1])])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            # V2, so coloca se o tamanho for maior que 1
            if len(new_ids) > 1:
                new_ids_counts[tuple(new_ids)] = freq
        return new_ids_counts

    def batch_encode(self, texts, num_workers=None):
        """
        Encoda uma lista de textos em paralelo.
        Retorna uma lista de listas de tokens na mesma ordem.
        """
        # Fixamos o dicionário de merges para passar para os processos
        encode_with_merges = partial(_encode_single_text, merges=self.merges)

        # O num_workers padrão é o número de CPUs do seu computador
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # O executor.map garante que a ordem dos resultados seja a mesma do input
            # Envolvemos o map com tqdm. O list() força a execução e atualiza a barra.
            results = list(
                tqdm(
                    executor.map(encode_with_merges, texts),
                    total=len(texts),
                    desc="Tokenizando em paralelo",
                    unit="doc",
                )
            )

        return results

    def encode(self, text):
        # Começamos com a sequência de bytes brutos
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self.get_stats([tokens])
            # Crucial: encontrar o par com o menor ID de merge (maior prioridade)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # Se o par mais "prioritário" não está nos merges, terminamos
            if pair not in self.merges:
                break

            # Aplica o merge e repete o processo
            new_id = self.merges[pair]
            tokens = self.merge(tokens, pair, new_id)

        return tokens

    def decode(self, ids):
        # Concatena os bytes de cada token ID
        tokens_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Transforma de volta em string (UTF-8)
        return tokens_bytes.decode("utf-8", errors="replace")

    def train(self, texts, vocab_size):
        # 1. Pré-tokenização
        print("Pré-processando textos...")
        words_counts = Counter()
        for t in texts:
            words_counts.update(re.findall(SPLIT_PATTERN, t))

        ids_counts = {
            tuple(w.encode("utf-8")): freq for w, freq in words_counts.items()
        }

        self.merges = {}
        num_merges = vocab_size - 256
        i = 0

        # Inicializa o tqdm manualmente com o total de merges esperados
        with tqdm(total=num_merges, desc="Treinando BPE", unit="merge") as pbar:
            while i < num_merges:
                stats = self.get_stats_unique(ids_counts)
                if not stats:
                    break

                # Seleção de top k pares sem overlap
                selected_for_merge = []
                for pair, freq in sorted(
                    stats.items(), key=lambda item: item[1], reverse=True
                ):
                    if all(
                        pair[0] not in p and pair[1] not in p
                        for p in selected_for_merge
                    ):
                        selected_for_merge.append(pair)
                    else:
                        break

                    # if len(selected_for_merge) >= 100:
                    #     break

                # Guarda o valor inicial de i para calcular o delta de progresso
                start_i = i
                to_merge = {}
                for top_pair in selected_for_merge:
                    if i >= num_merges:
                        break

                    new_id = 256 + i
                    to_merge[top_pair] = new_id
                    self.merges[top_pair] = new_id
                    i += 1
                ids_counts = self.merge_ids(ids_counts, to_merge)

                # Atualiza a barra de progresso com o número real de merges feitos nesta rodada
                pbar.update(i - start_i)

                # Opcional: Mostra a frequência do par mais forte no postfix
                if selected_for_merge:
                    top_freq = stats[selected_for_merge[0]]
                    pbar.set_postfix({"max_freq": top_freq, "vocab": 256 + i})

        self.vocab = self.build_vocab(self.merges)

    def print_merges(self):
        for (p1, p2), new_id in self.merges.items():
            print(f"({p1}, {p2}) -> {self.decode([new_id])}")

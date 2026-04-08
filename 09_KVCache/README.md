# 09_KV_Cache

<a href="https://youtu.be/Wv5c8gTnVN4" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto implementamos o **KV Cache** do zero — uma das otimizações mais importantes para a inferência de Transformers autorregressivos. Partindo do GPT implementado no projeto 07, adicionamos cache de Keys e Values na atenção multi-cabeça, eliminando o recomputo redundante desses tensores a cada passo da geração.

**O resultado: saímos de 90 tokens/s para 210 tokens/s — um ganho de ~2.4×, sem nenhuma mudança nos pesos do modelo.**

![Comparação de velocidade](./media/comparacao.gif)

**Tópicos abordados:**
- Por que o decoder autoreggressivo recomputa K e V desnecessariamente a cada passo
- Implementação da classe `KVCache` — armazena Keys e Values entre os passos de geração
- `CachedQKVMultiheadAttention` — atenção que usa o cache no `forward`
- `CachedDecoder` / `CachedDecoderLM` — decoder completo com cache e rastreamento de posição
- Estratégia de geração: primeiro passo processa o prompt inteiro; os seguintes enviam apenas 1 token
- Comparação direta: `generate.py` (sem cache) vs `generate_with_cache.py` (com cache)

---

## 📁 Estrutura do projeto

```
09_KVCache/
├── src/
│   ├── model.py        # DecoderLM (sem cache) + CachedDecoderLM (com cache)
│   ├── transformer.py  # FeedForward, Positional Encoding e KVCache
│   ├── attention.py    # QKVMultiheadAttention + CachedQKVMultiheadAttention
│   ├── dataset.py      # Dataset para geração de linguagem
│   ├── scheduler.py    # Linear warmup scheduler
│   └── tokenizer.py    # BPETokenizer e tokens especiais
├── artifacts/
│   └── bpe_tokenizer_100k.json   # Merges do tokenizador BPE (100k merges)
├── data/
│   └── links.txt                 # Links dos artigos baixados da Wikipedia
├── media/
│   └── comparacao.gif            # Demo comparando geração com e sem cache
├── tokenize_dataset.py           # Pré-tokeniza o dataset para acelerar o treinamento
├── train_decoder_wiki.py         # Script de treinamento (idêntico ao do 07_GPT)
├── generate.py                   # Geração sem cache (baseline)
├── generate_with_cache.py        # Geração com KV Cache
└── requirements.txt
```

---

## 🧠 Como o KV Cache funciona

Em um Transformer *decoder-only*, a geração é **autorregressiva**: a cada passo, o modelo recebe **toda a sequência gerada até agora** e prevê apenas o próximo token.

```
Passo 1: Forward([t1])          → prevê t2
Passo 2: Forward([t1, t2])      → prevê t3
Passo 3: Forward([t1, t2, t3])  → prevê t4
...
```

O problema é que, a cada passo, Keys e Values de **todos os tokens anteriores** são recomputados do zero — trabalho completamente redundante.

O KV Cache resolve isso **armazenando os K e V já computados** e apenas adicionando o novo token a cada passo:

```
Passo 1: Forward([t1])     → computa K,V de t1 → cache: {K:[t1], V:[t1]}
Passo 2: Forward([t2])     → computa K,V de t2 → cache: {K:[t1,t2], V:[t1,t2]}
Passo 3: Forward([t3])     → computa K,V de t3 → cache: {K:[t1,t2,t3], V:[t1,t2,t3]}
```

A Query continua sendo computada só para o token atual, mas a atenção usa **todos os K e V acumulados** — o resultado é idêntico, mas com uma fração do custo.

### Estratégia de geração com cache

```
1. Primeiro passo  → processa o prompt completo de uma vez → popula o cache
2. Passos seguintes → envia apenas o último token gerado → apenas 1 novo K,V por camada
```

---

## 🏗️ Arquitetura

### `KVCache` (`src/transformer.py`)

Estrutura simples que acumula os tensores de Keys e Values ao longo dos passos:

```python
class KVCache:
    def add(self, keys, values):
        # concatena os novos K,V com os do cache
        # descarta os mais antigos se ultrapassar max_len
        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
```

### `CachedQKVMultiheadAttention` (`src/attention.py`)

Atenção multi-cabeça com cache integrado. A única diferença em relação à versão original é que, no `forward`, os K e V gerados para o token atual são adicionados ao cache antes do cálculo da atenção:

```python
k, v = self.cache.add(k, v)   # k e v agora contêm toda a sequência
att = QKVAttention(q, k, v, mask)
```

### `CachedDecoderLM` (`src/model.py`)

Usa `CachedDecoder` (com `CacheTransformerBlock`) em vez do `Decoder` original. O `generate` built-in:
- Reseta o cache antes de cada geração
- No primeiro passo, processa o prompt inteiro e registra a posição atual
- Nos passos seguintes, passa apenas o último token + a posição correta para o Positional Encoding

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

> Para usar GPU (recomendado), o `requirements.txt` já aponta para os binários CUDA do PyTorch.

### 2. Prepare os dados e pré-tokenize

*(Igual ao projeto 07_GPT — os pesos treinados lá podem ser reutilizados aqui.)*

```bash
python tokenize_dataset.py
```

### 3. Treine o modelo

```bash
python train_decoder_wiki.py
```

### 4. Gere texto

**Sem cache (baseline):**
```bash
python generate.py
```

**Com KV Cache:**
```bash
python generate_with_cache.py
```

### Comparação de velocidade

| Modo | Tokens/s |
| :--- | ---: |
| Sem KV Cache | ~89 |
| **Com KV Cache** | **~214** |

> Ganho de ~**2.4×** na velocidade de geração, usando exatamente os mesmos pesos.

---

## 🛠️ Pré-requisitos

- Python 3.12+
- PyTorch 2.x (CUDA recomendado para treinamento)
- GPU com ≥ 8GB VRAM para experimentos confortáveis

---

## 📦 Dataset utilizado

* [Wikipedia PT — Wikimedia Dumps](https://dumps.wikimedia.org/ptwiki/)
* [Concatenado de avaliações PT-BR](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets?select=concatenated.csv)
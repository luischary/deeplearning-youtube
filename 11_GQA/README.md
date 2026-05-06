# 11 - GQA: Grouped Query Attention

> **Como reduzir o consumo do KV Cache em 4x ou mais sem perder performance.**

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20aula-red?logo=youtube)](https://youtu.be/ZOandvv43uk)
[![Paper](https://img.shields.io/badge/Paper-GQA%20(2023)-blue?logo=arxiv)](https://arxiv.org/pdf/2305.13245)

---

## 🎯 O Problema

Modelos com KV Cache consomem memória de forma linear conforme o contexto cresce. A conta é simples:

$$\text{Memória}_{KV} = L_{seq} \times n_{heads} \times d_{head} \times n_{layers} \times 2 \times 4 \text{ bytes}$$

Para um modelo pequeno (12 heads, 64 dims/head, 8 layers), isso dá ~47 MB por mil tokens — razoável. Já para o **Mistral 7B** (32 heads, 128 dims/head, 32 layers), o custo sobe para **~1 GB por mil tokens**. Mil tokens é praticamente uma página de documento, o que limita bastante o uso prático de contextos longos.

A raiz do problema é que no Multi-Head Attention tradicional, **cada head guarda seu próprio par K/V no cache**. Mas será que realmente precisamos de tantos pares K/V assim?

---

## 💡 A Solução: Grouped Query Attention

O paper propõe um espectro de soluções ao variar o número de heads de K e V:

### Multi-Query Attention (MQA) — o extremo
Todas as queries compartilham **um único par K/V**. É a redução máxima de memória, mas o modelo perde capacidade expressiva: todas as heads de query precisam "concordar" com a mesma representação de contexto. Na prática, a performance cai de forma considerável.

### Grouped Query Attention (GQA) — o equilíbrio
As queries são divididas em **grupos**, e cada grupo compartilha um par K/V. Com grupos de tamanho 4, por exemplo:

- A memória do KV Cache cai **4 vezes**
- A queda de performance é de apenas **~0.1 ponto** nas métricas do paper
- O tempo de inferência equipara-se ao MQA

Isso torna o GQA o padrão de fato em LLMs modernos. O Mistral 7B, por exemplo, usa GQA com grupos de 8 — derrubando o custo de 1 GB para ~125 MB por mil tokens.

A intuição é que **queries precisam de diversidade** (pontos de vista diferentes para buscar informação), mas **keys e values podem ser compartilhados** sem grande perda, pois o que importa é a variedade de "perguntas", não de "respostas".

---

## 🏗️ Implementação em PyTorch

### Projeções menores para K e V

A principal mudança em relação ao Multi-Head Attention normal está nas projeções lineares:

```python
self.proj_q = nn.Linear(d_model, d_model)                          # todas as heads
self.proj_k = nn.Linear(d_model, int(self.n_groups * self.head_dim))  # apenas n_groups
self.proj_v = nn.Linear(d_model, int(self.n_groups * self.head_dim))  # apenas n_groups
```

### `repeat_interleave` para compatibilidade

Durante o treinamento sem kernel CUDA especializado, expandimos K e V de volta ao tamanho de Q usando `repeat_interleave` (e **não** `repeat`, que teria ordem errada):

```python
# group_size = n_heads // n_groups
k = k.repeat_interleave(self.n_heads // self.n_groups, dim=1)
v = v.repeat_interleave(self.n_heads // self.n_groups, dim=1)
```

O resultado: `[1, 1, 2, 2, 3, 3, ...]` em vez de `[1, 2, 3, 1, 2, 3, ...]`. Isso garante que cada grupo de queries veja o par K/V correto.

### No KV Cache, o ganho de memória é real

Durante a inferência com cache, os K/V são armazenados no **tamanho reduzido** (n_groups × head_dim). O `repeat_interleave` ocorre apenas no momento do cálculo da atenção, não no armazenamento.

### Configuração flexível no modelo

O `TransformerBlock` aceita um parâmetro `att_group_size`:

```python
# att_group_size <= 0  →  Multi-Head Attention normal
# att_group_size > 0   →  Grouped Query Attention
TransformerBlock(embed_dim=768, n_heads=12, hidden_size=3072, att_group_size=4)
```

---

## 📂 Estrutura do Código

```
11_GQA/
├── src/
│   ├── attention.py     # GQMultiheadAttention e CachedGQMultiheadAttention
│   ├── model.py         # TransformerBlock com suporte a att_group_size
│   ├── rotary.py        # RoPE (Rotary Positional Encoding)
│   ├── transformer.py   # FeedForward, KVCache e blocos base
│   ├── dataset.py       # Dataset para treinamento
│   ├── scheduler.py     # Learning rate scheduler
│   └── tokenizer.py     # Tokenizador BPE
├── generate_with_cache.py  # Geração de texto com KV Cache + GQA
└── requirements.txt
```

### Arquivos-chave

**`src/attention.py`** — Implementações de atenção:
- `GQMultiheadAttention`: atenção agrupada para treinamento, usando `repeat_interleave` para expandir K/V.
- `CachedGQMultiheadAttention`: versão com KV Cache para inferência, onde K/V são armazenados compactados.
- `grouped_query_attention`: função standalone com broadcasting nativo (sem `repeat_interleave`) usando `unsqueeze` e `torch.matmul`.

**`src/model.py`** — `TransformerBlock` com lógica condicional: usa `GQMultiheadAttention` quando `att_group_size > 0`, ou `QKVMultiheadAttention` caso contrário.

---

## 🚀 Como usar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Geração de texto

```bash
python generate_with_cache.py
```

> Os artefatos necessários (modelo treinado e tokenizador) devem estar na pasta `artifacts/`.

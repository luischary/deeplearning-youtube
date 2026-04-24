# 10 - RoPE: Rotary Positional Encoding

> **Como girar vetores faz os modelos de linguagem modernos entenderem o contexto.**

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20aula-red?logo=youtube)](https://youtu.be/ehbooXr2PKg)
[![Paper](https://img.shields.io/badge/Paper-RoFormer%20(2021)-blue?logo=arxiv)](https://arxiv.org/pdf/2104.09864)

---

## 🎯 O Problema

Transformers são cegos para a ordem das palavras. Sem algum mecanismo de posição, o modelo lê uma sequência de tokens como um "saco de palavras" — a frase *"o cachorro mordeu o João"* seria indistinguível de *"o João mordeu o cachorro"*.

A solução original, proposta no paper *Attention Is All You Need*, introduz um **Positional Encoding absoluto**: um vetor fixo somado ao embedding de cada token de acordo com sua posição. Apesar de funcionar, esse método tem dois problemas sérios:

1. **Extrapolação ruim**: o modelo nunca viu posições além do contexto de treinamento, então quebra ao tentar processar sequências mais longas.
2. **Representação confusa**: a mesma palavra em posições diferentes recebe embeddings diferentes, forçando o modelo a gastar capacidade aprendendo que elas significam a mesma coisa.

A raiz do problema é que a posição é tratada de forma **absoluta**. A grande virada foi pensar em posição de forma **relativa**: ao invés de dizer "você está na posição 47", dizer "você está 3 tokens à frente do token anterior".

---

## 💡 A Ideia do RoPE

O **Rotary Positional Encoding (RoPE)**, introduzido no paper *RoFormer* em 2021, resolve isso com uma sacada elegante:

> **Codificar a posição como uma rotação no espaço vetorial.**

A posição não é mais somada ao embedding — ela é **incorporada como um ângulo de rotação** aplicado diretamente às queries (`q`) e keys (`k`) da camada de atenção. Isso garante que, no produto interno `q · k` (que é o coração da atenção), o que o modelo vê é naturalmente a **distância relativa** entre os tokens, não suas posições absolutas.

### Por que funciona?

A atenção entre dois tokens é calculada como:

$$\text{score}(q_m, k_n) = q_m \cdot k_n$$

Se rotacionarmos $q_m$ por um ângulo $m\theta$ e $k_n$ por $n\theta$, o produto escalar entre eles passa a depender apenas de $(m - n)\theta$ — ou seja, da **diferença de posições**, não das posições em si. Isso é posicionamento relativo implementado de forma implícita e eficiente.

### A Matemática (sem susto)

A rotação é feita no plano complexo usando a **identidade de Euler**:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

Ao invés de manter uma grande matriz de rotação esparsa, o vetor de embedding é visto como um número complexo e a rotação é aplicada com uma simples multiplicação:

$$\tilde{q}_m = q_m \cdot e^{im\theta}$$

Isso é eficiente (sem matrizes grandes), não altera a **norma** do vetor (só gira, não distorce) e é naturalmente compatível com o produto interno da atenção.

Para vetores de alta dimensão, as dimensões são agrupadas em pares, e cada par é rotacionado com uma frequência $\theta_k$ diferente:

$$\theta_k = \frac{1}{10000^{2k/d}}$$

Essa escala de frequências (idêntica à do Positional Encoding original) garante que diferentes dimensões codifiquem posição em diferentes "escalas de tempo".

---

## 🏗️ Onde o RoPE é Aplicado

Diferente do Positional Encoding tradicional, que modifica os embeddings de entrada, o RoPE é aplicado **dentro da camada de atenção**, diretamente nas queries e keys — e nunca nos values. A posição só é relevante para decidir *quem presta atenção em quem*, não para o conteúdo que é agregado.

```
Entrada → Embedding → [sem positional encoding aqui]
                              ↓
                       TransformerBlock
                              ↓
                    q, k, v = proj(x)
                              ↓
               RoPE aplicado em q e k ←── freqs pré-computadas
                              ↓
                     Atenção(q_rot, k_rot, v)
```

---

## 📂 Estrutura do Código

```
10_RoPE/
├── src/
│   ├── rotary.py        # Implementação central do RoPE
│   ├── attention.py     # Multi-head attention com suporte a RoPE
│   ├── model.py         # Arquitetura Decoder (GPT-like) com RoPE
│   ├── transformer.py   # Blocos base (FeedForward, KVCache, etc.)
│   ├── dataset.py       # Dataset para treinamento
│   ├── scheduler.py     # Learning rate scheduler
│   └── tokenizer.py     # Tokenizador BPE
├── generate_with_cache.py  # Geração de texto com KV Cache + RoPE
├── utils.py             # Utilitários gerais
└── requirements.txt
```

### Arquivos-chave

**`src/rotary.py`** — O coração do RoPE:
- `precompute_freqs_cis`: pré-computa as frequências complexas $e^{im\theta_k}$ para todas as posições e dimensões.
- `apply_rotary_emb`: aplica a rotação nas queries e keys usando multiplicação complexa.
- `apply_cross_rotary_emb`: variante para quando queries e keys têm comprimentos diferentes (ex: cross-attention).

**`src/attention.py`** — A atenção modificada recebe as frequências pré-computadas e aplica o RoPE antes de calcular os scores.

---

## 🚀 Como Executar

**1. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**2. Geração de texto com KV Cache:**
```bash
python generate_with_cache.py
```

---

## ✅ Vantagens do RoPE

| | Positional Encoding Absoluto | RoPE |
|---|---|---|
| **Extrapolação** | ❌ Quebra fora do contexto de treino | ✅ Generaliza naturalmente |
| **Posicionamento** | Absoluto | Relativo (implícito) |
| **Onde é aplicado** | Embeddings de entrada | Queries e Keys na atenção |
| **Altera a norma?** | Sim (soma vetores) | Não (apenas rotaciona) |
| **Usado em** | Transformer original | Llama, Mistral, Qwen, ... |

---

## 📚 Referências

- **Paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864) — Su et al. (2021)
- **Aula anterior (KV Cache)**: [`09_KVCache`](../09_KVCache/)

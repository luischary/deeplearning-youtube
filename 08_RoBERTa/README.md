# 08_RoBERTa

<a href="https://youtu.be/TJxAQc1EEZ0" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto implementamos um **Transformer Encoder do zero** e o treinamos utilizando a técnica de **Masked Language Modeling (MLM)** — a mesma estratégia de pré-treinamento do BERT/RoBERTa. Em seguida, aplicamos o modelo em uma tarefa de **classificação de texto** na base IMDB e comparamos dois cenários:

1. **Treinamento do zero** — o classificador é treinado diretamente na base IMDB, sem qualquer pré-treinamento.
2. **Fine-tuning com modelo pré-treinado** — o encoder é pré-treinado via MLM na Wikipedia em inglês e depois ajustado (fine-tuned) na base IMDB.

Os resultados mostram claramente o ganho obtido com o pré-treinamento:

| Cenário | Acurácia (teste) | F1 |
| :--- | :---: | :---: |
| Treinado do zero | 82% | 0.8285 |
| Pré-treinado — 100k steps (Wikipedia EN) | 86% | 0.8608 |
| Pré-treinado — 200k steps (Wikipedia EN) | 88% | 0.8781 |

**Tópicos abordados:**
- Treinamento do tokenizador BPE na Wikipedia em inglês — [train_tokenizer.py](./train_tokenizer.py)
- Pré-tokenização do corpus para acelerar o pré-treinamento — [tokenize_dataset.py](./tokenize_dataset.py)
- Implementação do Encoder (TransformerBlock + MLM head) — [src/model.py](./src/model.py)
- Pré-treinamento via **Masked Language Modeling** — [train_encoder.py](./train_encoder.py)
- Fine-tuning para classificação binária (IMDB) — [train_encoder.py](./train_encoder.py)
- Avaliação e comparação dos modelos — [eval_imdb.py](./eval_imdb.py)
- Inferência com máscara (teste do MLM) — [test_encoder.py](./test_encoder.py)

---

## 📁 Estrutura do projeto

```
08_RoBERTa/
├── src/
│   ├── model.py        # Encoder, EncoderMLM e EncoderForClassification
│   ├── transformer.py  # FeedForward e Positional Encoding senoidal
│   ├── attention.py    # Multi-Head Self-Attention
│   ├── dataset.py      # DatasetMLM, DatasetMLMTokenized e DatasetClassificacao
│   ├── scheduler.py    # Linear warmup scheduler
│   └── tokenizer.py    # BPETokenizer e tokens especiais
├── artifacts/
│   ├── bpe_tokenizer_100k.json       # Merges BPE (100k merges, base PT)
│   └── tokenizer_wiki_en_5k.json     # Tokenizador BPE (5k vocab, Wikipedia EN)
├── data/
│   ├── dados_utilizados.txt          # Descrição dos dados
│   ├── prepara.ipynb                 # Preparação dos dados IMDB
│   └── verifica_erros.ipynb          # Verificação de erros no dataset
├── train_tokenizer.py   # Treina o tokenizador BPE na Wikipedia EN
├── tokenize_dataset.py  # Pré-tokeniza o corpus e salva partições .npy
├── train_encoder.py     # Pré-treinamento MLM e fine-tuning para classificação
├── eval_imdb.py         # Avaliação dos modelos no conjunto de teste IMDB
├── inference_imbdb.py   # Inferência com o melhor modelo pré-treinado
├── test_encoder.py      # Testa o MLM: prediz tokens mascarados
├── utils.py             # CheckpointHandler e utilitários de carregamento
├── metricas.txt         # Resultados dos experimentos
└── requirements.txt
```

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Prepare os dados

Execute os notebooks para baixar e processar as bases:

- `data/prepara.ipynb` — processa a base IMDB e salva em `data/imdb/base_imdb.pq`

### 3. Treine o tokenizador BPE

Treina um tokenizador BPE com vocabulário de 5.000 tokens em uma amostra da Wikipedia em inglês:

```bash
python train_tokenizer.py
```

O tokenizador treinado é salvo em `artifacts/tokenizer_wiki_en_5k.json`.

### 4. Pré-tokenize o corpus para MLM

Tokeniza o corpus da Wikipedia e salva em partições `.npy` para acelerar o pré-treinamento:

```bash
python tokenize_dataset.py
```

### 5. Pré-treine o Encoder com Masked Language Modeling

```bash
python train_encoder.py
```

O script treina o `EncoderMLM` usando mascaramento aleatório de 15% dos tokens (mesmo esquema do BERT). Os checkpoints são salvos em `modelos_treinados/` e as métricas são registradas no TensorBoard.

```bash
tensorboard --logdir modelos_treinados/
```

### 6. Fine-tuning para classificação (IMDB)

No mesmo `train_encoder.py`, ajuste as configurações para o modo de classificação — carregando os pesos pré-treinados no `EncoderForClassification` e treinando na base IMDB.

### 7. Avalie os modelos

```bash
python eval_imdb.py
```

Gera o `classification_report` do sklearn no conjunto de teste e encontra o melhor threshold para maximizar o F1-score.

---

## 🛠️ Arquitetura do modelo

| Hiperparâmetro | Valor |
| :--- | :---: |
| `vocab_size` | 5.000 |
| `embed_dim` | 256 |
| `num_heads` | 8 |
| `hidden_size` | 1.024 |
| `num_layers` | 4 |
| `max_len` | 512 |
| `dropout` | 0.1 |

---

## 📚 Referências

- Devlin et al. (2018) — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Liu et al. (2019) — [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- Vaswani et al. (2017) — [Attention is All You Need](https://arxiv.org/abs/1706.03762)

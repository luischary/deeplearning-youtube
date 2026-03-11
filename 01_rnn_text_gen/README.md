# 01_rnn_text_gen

<a href="https://youtu.be/zmaNIXOWbuo" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto implementamos uma rede neural recorrente (RNN/LSTM) que aprende a **gerar texto** em português, treinada sobre reviews de e-commerce. O pipeline é construído do zero: da preparação dos dados até a geração com temperatura.

**Tópicos abordados:**
* Dados e preparação - [1_dataprep.ipynb](./1_dataprep.ipynb)
* Tokenização - [2_tokenize.ipynb](./2_tokenize.ipynb)
* Criação dos componentes (tokenizador, dataset e modelo) - [src](./src/)
* Loop de treinamento - [train.py](./train.py)
* Geração de texto com o modelo (amostragem simples e com temperatura) - [generate.py](./generate.py)

---

## 📁 Estrutura do projeto

```
01_rnn_text_gen/
├── src/
│   ├── model.py        # Definição da rede LSTM
│   ├── dataset.py      # Dataset e DataLoader
│   └── tokenizer.py    # Tokenizador baseado em vocabulário
├── artifacts/
│   ├── tokenize_dict.json    # Vocabulário token → id
│   └── detokenize_dict.json  # Vocabulário id → token
├── 1_dataprep.ipynb    # Preparação e exploração dos dados
├── 2_tokenize.ipynb    # Construção do tokenizador
├── train.py            # Script de treinamento
└── generate.py         # Geração de texto (greedy e com temperatura)
```

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

> Para usar GPU (recomendado), o `requirements.txt` já aponta para os binários CUDA do PyTorch.

### 2. Treine o modelo

```bash
python train.py
```

### 3. Gere texto

```bash
python generate.py
```

---

## 🛠️ Pré-requisitos

- Python 3.12+
- PyTorch 2.x (CPU ou CUDA)

---

## 📦 Datasets utilizados

* [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download&select=olist_order_reviews_dataset.csv)
* [Concatenado de avaliações PT-BR](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets?select=concatenated.csv)
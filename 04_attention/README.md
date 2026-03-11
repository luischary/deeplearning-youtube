# 04_attention

<a href="https://youtu.be/S1pgy9xNTjw" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto implementamos o mecanismo de **atenção QKV (Query, Key, Value)** utilizado na arquitetura Transformer, do zero com PyTorch. A atenção é o componente central dos modelos modernos de linguagem — entender sua matemática é essencial antes de estudar o Transformer completo.

Foram implementados os 3 tipos de atenção mais comuns:

- **Self-attention** — cada token atende a todos os outros da sequência (usado no Encoder)
- **Masked self-attention** — atenção com máscara causal, impedindo que tokens "vejam o futuro" (usado no Decoder)
- **Cross-attention** — queries de uma sequência atendem a keys/values de outra (conexão Encoder→Decoder)

> **Próximo passo:** o projeto [05_multihead_attention](../05_multihead_attention) estende este conteúdo com múltiplas cabeças de atenção em paralelo.

Veja o notebook [attention.ipynb](./attention.ipynb)

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Abra o notebook

```bash
jupyter notebook attention.ipynb
```

---

## 🛠️ Pré-requisitos

- Python 3.12+
- PyTorch, seaborn, matplotlib (ver `requirements.txt`)
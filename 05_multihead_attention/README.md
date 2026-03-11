# 05_multihead_attention

<a href="https://youtu.be/SwDXohxeC7g" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto evoluímos o mecanismo de atenção do projeto anterior para a versão com **múltiplas cabeças (Multi-Head Attention)**. Em vez de calcular uma única atenção, o modelo aprende várias representações em paralelo — cada "cabeça" pode focar em aspectos diferentes da sequência — e depois as concatena.

> **Pré-requisito:** o projeto [04_attention](../04_attention) cobre a atenção QKV de cabeça única que serve de base para este.

Foram implementados os 3 tipos de atenção mais comuns na versão multi-head:

- **Self-attention** — cada token atende a todos os outros da sequência (usado no Encoder)
- **Masked self-attention** — atenção com máscara causal, impedindo que tokens "vejam o futuro" (usado no Decoder)
- **Cross-attention** — queries de uma sequência atendem a keys/values de outra (conexão Encoder→Decoder)

> **Próximo passo:** o projeto [06_transformer](../06_transformer) usa este componente para construir a arquitetura Transformer completa.

Veja o notebook [multihead_attention.ipynb](./multihead_attention.ipynb)

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Abra o notebook

```bash
jupyter notebook multihead_attention.ipynb
```

---

## 🛠️ Pré-requisitos

- Python 3.12+
- PyTorch, seaborn, matplotlib (ver `requirements.txt`)
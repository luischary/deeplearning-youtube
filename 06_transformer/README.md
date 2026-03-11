# 06_transformer

<a href="https://youtu.be/uPg-1GUnVSE" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto

Neste projeto implementamos a arquitetura Transformer completa (Encoder e Decoder) **do zero**, utilizando apenas PyTorch. O código segue de perto o paper original [*Attention is All You Need* (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) e é pensado para fins didáticos — cada componente é implementado de forma explícita e comentada.

> **Pré-requisito recomendado:** o projeto [05_multihead_attention](../05_multihead_attention) cobre em detalhes o mecanismo de atenção multi-cabeça que é utilizado aqui.

### Componentes implementados

- **Positional Encoding** — embeddings senoidais para injetar informação de posição
- **Multi-Head Self-Attention** — atenção com múltiplas cabeças (self-attention e masked self-attention)
- **FeedForward** — rede densa com ativação ReLU e dropout
- **Conexões Residuais + Layer Norm** — estabilidade e convergência do treinamento
- **TransformerBlock** — bloco completo combinando atenção + FFN + normas
- **Encoder** — pilha de TransformerBlocks sem máscara causal
- **Decoder** — pilha de TransformerBlocks com máscara causal (geração autoregressiva)

---

## 📁 Estrutura do projeto

```
06_transformer/
├── src/
│   ├── attention.py      # QKVAttention, QKVMultiheadAttention e causal mask
│   ├── transformer.py    # Positional Encoding e FeedForward
│   └── model.py          # TransformerBlock, Encoder e Decoder
├── test.py               # Script de exemplo para testar Encoder e Decoder
└── requirements.txt
```

---

## 🚀 Como rodar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute o script de teste

```bash
python test.py
```

O script instancia um Encoder e um Decoder com parâmetros pequenos e imprime o shape da saída, confirmando que o pipeline funciona corretamente.

---

## 🛠️ Pré-requisitos

- Python 3.12+
- PyTorch (ver `requirements.txt`)

---

## 📚 Referências

- Vaswani et al. (2017) — [Attention is All You Need](https://arxiv.org/abs/1706.03762)
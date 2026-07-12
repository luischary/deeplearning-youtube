# 12 - Como Neurônios Viram Redes Neurais?

> **Entendendo por que funções de ativação são o ingrediente essencial de qualquer rede neural.**

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20aula-red?logo=youtube)](https://www.youtube.com/watch?v=JL35p8PE3jk)

---

## 🎯 O Problema

Um único neurônio é apenas uma combinação linear de entradas: $y = w_1 x_1 + w_2 x_2 + b$. E se empilharmos vários desses neurônios em camadas? O resultado ainda é uma combinação linear — duas matrizes multiplicadas equivalem a uma só. Por mais camadas que se adicione, a rede continua sendo equivalente a uma única transformação linear, incapaz de aprender qualquer padrão não-linear.

A prova prática está aqui: treine uma rede de duas camadas **sem** função de ativação em um problema com fronteira de decisão curva. Ela vai falhar.

---

## 💡 A Solução: Não-Linearidade

Funções de ativação como **ReLU** ($\max(0, x)$) inserem não-linearidade entre as camadas, quebrando o colapso matricial. Isso permite que a rede **aprenda representações complexas** e aproxime qualquer função contínua (Teorema da Aproximação Universal).

A diferença em termos de código é mínima — uma linha:

```python
# Rede Linear (falha em problemas não-lineares)
self.net = nn.Sequential(
    nn.Linear(2, 8),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Rede Não-Linear (resolve o problema)
self.net = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),        # <-- essa linha faz toda a diferença
    nn.Linear(8, 1),
    nn.Sigmoid()
)
```

O impacto visual é imediato ao observar a fronteira de decisão evoluindo durante o treinamento.

---

## 📂 Estrutura do Código

```
12_neuronio_vira_rede/
├── experimento_ativacao.py           # Animação da fronteira de decisão (problema das luas)
├── experimento_circulos.py           # Comparação estática (problema dos círculos)
├── evolucao_redes_pcolormesh.mp4     # Vídeo da animação gerado pelo experimento_ativacao.py
└── requirements.txt
```

### Arquivos

**`experimento_ativacao.py`** — Gera uma **animação** mostrando a fronteira de decisão das duas redes (linear vs ReLU) sendo atualizada época por época, treinando simultaneamente no dataset `make_moons`. O resultado é salvo como `evolucao_redes_pcolormesh.mp4`.

**`experimento_circulos.py`** — Versão estática. Treina as duas redes no dataset `make_circles` e plota os resultados lado a lado, deixando a diferença de capacidade expressiva visível de forma imediata.

---

## ▶️ Como Executar

**1. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**2. Experimento estático (círculos):**
```bash
python experimento_circulos.py
```
Abre uma janela com duas subplots: a rede linear incapaz de separar os círculos vs a rede com ReLU separando perfeitamente.

**3. Animação (luas):**
```bash
python experimento_ativacao.py
```
Roda o treinamento e salva `evolucao_redes_pcolormesh.mp4` na pasta. Também exibe a animação ao vivo se executado em ambiente com suporte a GUI.

---

## 🧠 Conceitos Abordados

- **Colapso linear:** por que empilhar camadas lineares não aumenta capacidade do modelo
- **Funções de ativação:** ReLU, Sigmoid e o papel de cada uma
- **Fronteira de decisão:** como visualizar o que uma rede está "aprendendo" no espaço 2D
- **Teorema da Aproximação Universal:** qualquer função contínua pode ser aproximada por uma rede com pelo menos uma camada oculta e ativação não-linear
- **Datasets clássicos de toy:** `make_moons` e `make_circles` do scikit-learn

---

## 🛠️ Dependências

| Biblioteca | Uso |
| :--- | :--- |
| `torch` | Definição e treinamento dos modelos |
| `matplotlib` | Visualização e geração da animação |
| `scikit-learn` | Geração dos datasets (`make_moons`, `make_circles`) |

# 13 - SGD: Como Gradientes Viram Aprendizado?

> **Do Gradiente Descendente ao Mini-Batch SGD: taxa de aprendizado, ruído de batch e otimização no MNIST.**

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20aula-red?logo=youtube)](https://youtu.be/yL9Y7kIqFNM)

---

## 🎯 O Problema

No vídeo anterior, acompanhamos o caminho completo do **Backpropagation** e obtivemos uma informação essencial: as derivadas parciais e os gradientes ($\nabla_\theta \mathcal{L}$), indicando exatamente quanto cada parâmetro da rede influencia no erro final.

Porém, existe um detalhe crucial: **o backpropagation não atualiza os pesos sozinho**. 
Saber como um peso afeta a loss não significa saber diretamente qual deve ser seu novo valor. Em problemas simples e lineares, é possível derivar soluções fechadas (como as equações normais da regressão linear). Mas em redes neurais profundas — com milhões de parâmetros e não-linearidades entrelaçadas —, uma solução analítica direta é inviável.

Precisamos de um **processo iterativo**: usar a informação local calculada a cada instante para dar passos graduais e contínuos em direção aos parâmetros que minimizam o erro do modelo.

---

## 💡 A Solução: Gradiente Descendente e Otimizadores

### 1. A Regra de Atualização

O gradiente aponta na direção de **maior crescimento** da função de perda. Portanto, para minimizar a perda, devemos nos mover no sentido oposto:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

Onde:
- $\theta$: vetor de parâmetros do modelo (pesos e vieses)
- $\eta$ (eta): taxa de aprendizado (*learning rate*)
- $\nabla_\theta \mathcal{L}$: gradiente da função de perda em relação aos parâmetros

### 2. O Papel do Learning Rate ($\eta$)

O *learning rate* é o hiperparâmetro mais fundamental do treinamento:
- **Pequeno demais:** passos minúsculos, convergência extremamente lenta e risco de ficar preso em platôs da loss.
- **Ideal:** descida consistente e rápida até a vizinhança do mínimo global/local aceitável.
- **Grande demais:** passos desproporcionais que saltam por cima do vale de menor perda, causando oscilações violentas ou até divergência matemática (loss explodindo para $\infty$ ou `NaN`).

### 3. De Full Batch a Mini-Batch SGD

Conforme os conjuntos de dados aumentam, a forma de calcular o gradiente precisa se adaptar:

| Abordagem | Amostras por Update | Vantagens | Desvantagens |
| :--- | :---: | :--- | :--- |
| **Full Batch GD** | Todo o dataset ($N$) | Gradiente exato, trajetória suave e estável | Custo computacional e de memória proibitivo para datasets grandes |
| **SGD Puro (Estocástico)** | 1 amostra | Atualizações imediatas e baixíssimo uso de memória | Altíssima variância e trajetória caótica ("zigue-zague") |
| **Mini-Batch SGD** | Lote fixo (ex: 32, 64, 128) | Vetorização eficiente em GPU, gradiente balanceado e ruído saudável | Introduz o hiperparâmetro de batch size |

O **Mini-Batch SGD** tornou-se o padrão da indústria porque oferece o melhor dos dois mundos: aproveita a aceleração matricial do hardware e introduz um ruído controlado que ajuda os otimizadores a escaparem de mínimos locais ruins e pontos de sela.

---

## 🔬 Experimentos e Resultados

Este subprojeto contém experimentos progressivos, do cálculo manual de derivadas até o treinamento de uma rede neural real no MNIST. Todos os resultados são medidos empiricamente e salvos em `outputs/`.

### 01. Regressão Linear por GD Manual (`scripts/01_regressao_gd_manual.py`)
Treina o modelo $y = wx + b$ em dados sintéticos gerados por $y = 2x + 3 + \epsilon$ calculando analiticamente:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum (wx_i + b - y_i)x_i \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum (wx_i + b - y_i)$$

- Partindo de $w=-1.0$ e $b=0.0$, o modelo atinge $w \approx 2.01$ e $b \approx 2.99$ em 100 passos, demonstrando a descida logarítmica da loss MSE.

### 02. Comparação de Learning Rates (`scripts/02_regressao_comparar_learning_rates.py`)
Compara três regimes de taxa de aprendizado sob as mesmas condições iniciais:
- $\eta = 0.005$ (Pequeno): após 100 passos, ainda longe do ótimo ($\text{MSE} \approx 3.75$).
- $\eta = 0.05$ (Ideal): convergência rápida e precisa ($\text{MSE} \approx 0.24$).
- $\eta = 0.90$ (Excessivo): passos gigantescos que provocam instabilidade numérica e divergência ($\text{loss} > 10^{11}$).

### 03. Ruído do Gradiente por Batch Size (`scripts/03_ruido_gradiente_por_batch_size.py`)
Com os parâmetros $w$ e $b$ congelados no mesmo ponto, sorteamos múltiplos gradientes com batch sizes variados ($1, 4, 16, 64, 100$):
- **Esperança matemática:** a média dos gradientes estocásticos coincide com o gradiente full batch ($\mathbb{E}[\nabla \mathcal{L}_{\text{batch}}] = \nabla \mathcal{L}_{\text{full}}$).
- **Variância:** a dispersão em relação ao gradiente exato reduz-se rapidamente com o aumento do batch size (distância média cai de $\approx 8.03$ com batch 1 para $\approx 0.00$ com full batch).

### 04. Treinamento de MLP no MNIST com SGD (`scripts/04_mnist_treino_sgd.py`)
Treina uma rede neural MLP ($784 \to 64 \to 10$) com ativação ReLU e `CrossEntropyLoss` usando `torch.optim.SGD` com batch size 64:
- Acompanhamento da loss estocástica por update e da média móvel.
- Em apenas 3 épocas no conjunto completo do MNIST, atinge **>95.3% de acurácia** no conjunto de teste.

### 05. Comparação de Learning Rates no MNIST (`scripts/05_mnist_comparar_learning_rates.py`)
Treina a mesma MLP com $\eta \in \{0.01, 0.1, 0.5, 1.0\}$, mantendo a mesma inicialização de pesos e a mesma sequência de batches:
- $\eta = 0.01$: convergência lenta ($\approx 90.1\%$ de acurácia em 3 épocas).
- $\eta = 0.10$ e $\eta = 0.50$: convergência rápida e excelente generalização ($\approx 95.3\% - 95.4\%$).
- $\eta = 1.00$: maior instabilidade nos batches iniciais e leve perda de acurácia final ($\approx 94.8\%$).

### 06. Grade de Previsões (`scripts/06_mnist_grade_previsoes.py`)
Carrega o checkpoint gerado no script 04 e plota uma grade visual com amostras reais do conjunto de teste, exibindo a imagem do dígito e a classe prevista pelo modelo treinado.

---

## 📂 Estrutura do Código

```
13_SGD/
├── scripts/
│   ├── 01_regressao_gd_manual.py                 # GD implementado do zero em regressão linear
│   ├── 02_regressao_comparar_learning_rates.py   # Visualização da convergência e divergência
│   ├── 03_ruido_gradiente_por_batch_size.py      # Estudo empírico da variância vs batch size
│   ├── 04_mnist_treino_sgd.py                    # MLP no MNIST com mini-batch SGD (PyTorch)
│   ├── 05_mnist_comparar_learning_rates.py       # Comparação de taxas de aprendizado no MNIST
│   ├── 06_mnist_grade_previsoes.py               # Visualização dos dígitos e previsões
│   └── smoke_regressao.sh                        # Teste rápido dos scripts de regressão
├── src/sgd_video_experiments/
│   ├── common.py                                 # Utilitários de seeds, I/O e matplotlib
│   ├── mnist.py                                  # Arquitetura MLP, loaders e rotina de treino
│   └── regression.py                             # Geração de dados sintéticos e funções de GD
├── outputs/                                      # Gráficos, métricas (JSON) e CSVs gerados
├── pyproject.toml                                # Configuração do projeto e dependências
├── requirements.txt                              # Dependências para instalação via pip
└── transcription.txt                             # Transcrição da aula em vídeo
```

---

## ▶️ Como Executar

### 1. Preparação do Ambiente

Você pode usar o [`uv`](https://docs.astral.sh/uv/) (recomendado) ou o `pip` tradicional:

**Opção A — Com `uv`:**
```bash
cd 13_SGD
uv sync
```

**Opção B — Com `venv` e `pip`:**
```bash
cd 13_SGD
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Executando os Experimentos

Todos os scripts aceitam `--seed` e `--output-dir`. Para rodar na sequência do vídeo:

```bash
# 1) Regressão linear com atualização manual de w e b
python scripts/01_regressao_gd_manual.py

# 2) Comparando learning rates (pequeno, ideal e excessivo)
python scripts/02_regressao_comparar_learning_rates.py

# 3) Dispersão e ruído do gradiente por batch size
python scripts/03_ruido_gradiente_por_batch_size.py

# 4) Treinamento da MLP no MNIST com mini-batch SGD
python scripts/04_mnist_treino_sgd.py

# 5) Comparação de learning rates no MNIST
python scripts/05_mnist_comparar_learning_rates.py

# 6) Grade visual com as predições do modelo
python scripts/06_mnist_grade_previsoes.py
```

> **Dica (Execução Rápida):** Para testar o pipeline do MNIST sem aguardar o treinamento completo, utilize a flag `--quick`:
> ```bash
> python scripts/04_mnist_treino_sgd.py --quick
> python scripts/05_mnist_comparar_learning_rates.py --quick
> python scripts/06_mnist_grade_previsoes.py --count 8
> ```

---

## 🧠 Conceitos Abordados

- **Gradiente vs Otimizador:** a diferença entre calcular a direção do declive (Backprop) e decidir como atualizar os pesos (Otimizador).
- **Taxa de Aprendizado ($\eta$):** impacto de passos pequenos, ideais e excessivamente grandes na superfície de erro.
- **Gradiente Descendente em Lote (Full Batch):** estabilidade teórica vs restrições práticas de memória e escalabilidade.
- **SGD Puro e Mini-Batch SGD:** o balanço entre eficiência computacional em GPUs e ruído estocástico na otimização.
- **Propriedade do Estimador Não-Viesado:** demonstração empírica de que $\mathbb{E}[\nabla \mathcal{L}_{\text{batch}}] = \nabla \mathcal{L}_{\text{full}}$.
- **Treinamento de Redes Neurais:** aplicação do `torch.optim.SGD` com `CrossEntropyLoss` em um problema real de visão computacional (MNIST).

---

## 🛠️ Tecnologias e Dependências

| Biblioteca | Versão Mínima | Uso |
| :--- | :---: | :--- |
| `torch` | $\ge 2.5$ | Criação do modelo MLP, autograd e otimizador `torch.optim.SGD` |
| `torchvision` | $\ge 0.20$ | Download e transformações do dataset MNIST |
| `numpy` | $\ge 2.0$ | Cálculos matriciais dos experimentos de regressão e dispersão |
| `matplotlib` | $\ge 3.9$ | Geração dos gráficos comparativos, trajetórias e grade de previsões |

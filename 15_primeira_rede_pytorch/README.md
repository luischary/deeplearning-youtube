# 15 - Treine sua Primeira Rede Neural com PyTorch: Do Código aos Resultados

> **Construindo o pipeline completo de Deep Learning no PyTorch do zero absoluto: metadados, Dataset, DataLoader, autograd, otimização com SGD, normalização de dados e diagnóstico visual época a época no MNIST.**

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20aula-red?logo=youtube)](https://youtu.be/Gs9_rKslisw)

---

## 🎯 Da Teoria Manual ao Framework

Na aula anterior, implementamos um neurônio do zero absoluto usando apenas NumPy e cálculo manual: definimos os dados, calculamos o *forward pass*, avaliamos o erro quadrático (MSE), derivamos as derivadas parciais na mão e atualizamos os parâmetros com a regra do gradiente descendente.

O objetivo daquele exercício foi desmistificar o que acontece "debaixo do capô". Mas quando saímos de uma reta simples com 2 parâmetros ($w$ e $b$) para um problema de visão computacional — como reconhecer dígitos manuscritos no dataset MNIST —, a derivação analítica manual para cada camada se torna inviável.

É aqui que entra o **PyTorch**. O mapa mental e a receita iterativa continuam **rigorosamente os mesmos**, mas o framework assume o trabalho pesado:

| Etapa Conceitual | Implementação Manual (Aula 14) | No PyTorch (Aula 15) |
| :--- | :--- | :--- |
| **Estrutura dos Dados** | Vetores NumPy (`np.array`) | Tensores com suporte a hardware (`torch.Tensor`) |
| **Acesso às Amostras** | Listas e matrizes em memória | Abstração padronizada com `torch.utils.data.Dataset` |
| **Estratégia de Batches** | Fatiamento manual de arrays | Iterador otimizado com `torch.utils.data.DataLoader` (`shuffle`, `batch_size`, `num_workers`) |
| **Arquitetura & Parâmetros** | Variáveis $w$ e $b$ avulsas | Classes herdando de `torch.nn.Module` (`nn.Linear`, `nn.Sequential`) |
| **Cálculo de Gradientes** | Derivadas analíticas deduzidas à mão | Diferenciação automática com Autograd (`loss.backward()`) |
| **Atualização dos Pesos** | Subtração manual $\theta \leftarrow \theta - \eta \cdot \nabla \mathcal{L}$ | Otimizadores nativos (`optimizer.step()`, `optimizer.zero_grad()`) |
| **Avaliação & Métricas** | Cálculo estático pós-treino | Avaliação periódica com `model.eval()` e `torch.no_grad()` |

---

## 💡 O Pipeline de Treinamento no PyTorch

O script de treinamento executa um fluxo modular e bem delimitado:

```mermaid
flowchart TD
    A[Imagens em Disco<br>data/MNIST/images/] -->|mapeamento.py| B[Metadados CSV<br>path e label]
    B --> C[MNISTDataset<br>src/dataset.py]
    C -->|Carrega imagem, converte em tensor, escala /255| D[DataLoader<br>batch_size=32, shuffle=True]
    D --> E[Modelo PyTorch<br>Flatten -> Linear -> ReLU -> Linear]
    E --> F[Logits<br>Shape: BATCH, 10]
    F --> G[Função de Perda<br>F.cross_entropy logits, labels]
    G --> H[Autograd<br>loss.backward]
    H --> I[Otimizador SGD<br>optimizer.step]
    I -->|Próximo Batch / Época| D
    E -.->|Validação periódica| J[evaluate<br>Loss, Acurácia e Confiança Softmax]
```

### 1. Mapeamento e Metadados
Ao invés de varrer pastas repetidamente a cada época, o script `mapeamento.py` varre o diretório de imagens apenas uma vez e gera arquivos CSV estruturados (`metadados_treino.csv` com 60.000 amostras e `metadados_teste.csv` com 10.000 amostras), armazenando o caminho do arquivo e o rótulo do dígito correspondente.

### 2. Dataset Customizado (`MNISTDataset`)
Herda de `torch.utils.data.Dataset` e cumpre o contrato essencial do PyTorch:
- `__len__`: informa a quantidade total de registros.
- `__getitem__`: dado um índice, lê o arquivo PNG com `torchvision.io.read_image`, converte a imagem para tensor float32, aplica a função de transformação de escala e retorna o par `(imagem, label)`.

### 3. DataLoader
Responsável por agrupar os exemplos individuais retornados pelo dataset em tensores de lote (*mini-batches*), embaralhar os dados de treino a cada época (`shuffle=True`) e paralelizar a leitura em segundo plano com múltiplos processos (`num_workers=4`, `persistent_workers=True`).

### 4. A Anatomia do Treinamento
Dentro do loop, cada iteração sobre um batch de dados executa os passos fundamentais:
```python
optimizer.zero_grad()           # 1. Zera gradientes acumulados da iteração anterior
logits = model(batch_imagens)   # 2. Forward pass (produz logits não normalizados)
loss = F.cross_entropy(logits, batch_labels)  # 3. Calcula a perda (Cross-Entropy com Softmax interno)
loss.backward()                 # 4. Backpropagation (autograd preenche .grad de cada parâmetro)
optimizer.step()                # 5. Atualiza os parâmetros na direção oposta ao gradiente
```

---

## 🔬 A Jornada Experimental: Da Escala à Não-Linearidade

O projeto demonstra como três decisões simples de engenharia impactam drasticamente a estabilidade e a acurácia do modelo no conjunto de teste (10.000 imagens).

### 1. Modelo Linear sem Normalização (`linear`)
- **Arquitetura:** `nn.Flatten()` seguido de `nn.Linear(28 * 28, 10)`.
- **Entrada:** pixels brutos no intervalo $[0, 255]$.
- **Problema:** entradas em escala alta provocam logits desproporcionais, gerando valores de perda gigantescos ($\approx 1584$) e gradientes com magnitudes instáveis.
- **Resultado:** o modelo atinge $\approx 87.29\%$ de acurácia, mas a loss flutua sem suavidade.

### 2. O Impacto da Escala dos Dados (`linear_transform`)
- **Arquitetura:** a mesma camada linear (`784 -> 10`), sem alterar nenhum parâmetro da rede.
- **Entrada:** pixels divididos por $255.0$, limitando a escala ao intervalo $[0.0, 1.0]$.
- **Efeito:** as grandezas numéricas se estabilizam, os logits operam dentro de valores controlados e a loss inicial cai de dezenas para a casa de $2.29$, convergindo suavemente para $0.28$.
- **Resultado:** apenas mudando a escala de entrada, a acurácia salta para **$92.13\%$**.

### 3. Introduzindo Não-Linearidade (`mlp`)
- **Arquitetura:** Multilayer Perceptron com camada oculta e ativação não-linear:
  $$\text{Input } (28 \times 28) \longrightarrow \text{Flatten} \longrightarrow \text{Linear}(784, 64) \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(64, 10)$$
- **Entrada:** pixels normalizados em $[0.0, 1.0]$.
- **Efeito:** a adição da função de ativação ReLU quebra a linearidade do espaço vetorial, permitindo que a rede desenhe fronteiras de decisão complexas e curvas entre os dígitos.
- **Resultado:** a acurácia salta para **$96.51\%$** em apenas 5 épocas.

### Comparativo Consolidado

| Modelo | Normalização | Arquitetura | Loss Val (Época 0) | Loss Val (Época 5) | Acurácia Teste | Diagnóstico |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Linear Bruto** | Não ($0 - 255$) | `Linear(784, 10)` | $75.92$ | $1273.07$ | $87.29\%$ | Loss instável, gradientes descalibrados por entradas altas |
| **Linear Normalizado** | Sim ($0.0 - 1.0$) | `Linear(784, 10)` | $2.29$ | $0.28$ | $92.13\%$ | Convergência suave e salto imediato de $+4.8\%$ de acurácia |
| **MLP (Não-Linear)** | Sim ($0.0 - 1.0$) | `Linear(784, 64) + ReLU + Linear(64, 10)` | $2.30$ | **$0.12$** | **$96.51\%$** | Fronteiras expressivas, alta precisão e forte calibração |

---

## 📊 Diagnóstico e Inspeção Visual

Para além dos números agregados, o script `metricas.py` gera artefatos de diagnóstico para auditar o aprendizado do modelo:

1. **Evolução Época a Época:**
   - **Época 0 (Antes do Treino):** acurácia de $\approx 10.8\%$, compatível com o chute puramente aleatório entre 10 classes.
   - **Época 1:** a acurácia salta para $>92\%$, mas a rede ainda comete erros com alta confiança em casos difíceis.
   - **Épocas 2 a 5:** refinamento das probabilidades e aumento da confiança nas previsões corretas.

2. **Grades Visuais de Previsões (`grade_previsoes_epoca_X.png`):**
   - Amostras de teste inspecionadas individualmente.
   - Indicação em **verde** para predições corretas e **vermelho** para erros.
   - Exibição da probabilidade estimada via Softmax ($\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$) para auditar o nível de certeza do modelo.

3. **Curvas de Loss e Acurácia (`evolucao_metricas.png` e `zoom_test.png`):**
   - Comparação da curva de perda de treinamento vs validação, permitindo checar se o modelo está convergindo de forma equilibrada sem overfitting precoce.

4. **Relatório de Classificação Detalhado (`classification_reports.txt`):**
   - Métricas de Precision, Recall e F1-Score discriminadas por dígito ($0$ a $9$) em cada época de treinamento.

---

## 📂 Estrutura do Código

```
15_primeira_rede_pytorch/
├── data/
│   ├── metadados_treino.csv      # Mapeamento pré-computado das 60.000 imagens de treino
│   ├── metadados_teste.csv       # Mapeamento pré-computado das 10.000 imagens de teste
│   └── MNIST/images/             # Imagens em PNG separadas por pastas de dígitos (0 a 9)
├── modelos_treinados/
│   ├── linear/                   # Histórico e métricas do classificador linear sem escala
│   ├── linear_transform/         # Histórico e métricas do classificador linear com escala
│   └── mlp/                      # Histórico, métricas e grades de previsões do MLP
│       ├── historico.json
│       ├── historico_teste.json
│       └── metricas/
│           ├── classification_reports.txt
│           ├── evolucao_metricas.png
│           ├── zoom_test.png
│           └── grade_previsoes_epoca_*.png
├── src/
│   └── dataset.py                # Implementação customizada de MNISTDataset e default_transform
├── mapeamento.py                 # Varre as pastas de imagens e constrói os CSVs de metadados
├── train.py                      # Pipeline completo: Modelo, DataLoaders, loop de treino e evaluate
├── test.py                       # Script rápido para verificação de shapes e sanidade dos tensores
├── metricas.py                   # Gera gráficos de evolução, relatórios e grades visuais de dígitos
├── pyproject.toml                # Definição do ambiente e dependências via uv
└── transcription.txt             # Transcrição completa em áudio da aula do vídeo
```

---

## 🚀 Como Executar

### 1. Configurar o Ambiente

Você pode usar o gerenciador **uv** (recomendado) ou ambiente virtual padrão com `pip`:

**Usando uv:**
```bash
uv sync
```

**Usando pip tradicional:**
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install torch torchvision pandas pyarrow matplotlib scikit-learn tqdm
```

### 2. Gerar os Metadados das Imagens
```bash
python mapeamento.py
```

### 3. Treinar o Modelo
```bash
python train.py
```

### 4. Gerar Gráficos e Diagnósticos Visuais
```bash
python metricas.py
```
Os relatórios e as imagens serão salvos automaticamente em `modelos_treinados/<nome_modelo>/metricas/`.

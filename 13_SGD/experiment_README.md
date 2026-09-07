# Experimentos — Como gradientes viram aprendizado?

Subprojeto autocontido e reproduzível para gravar os experimentos do vídeo **“Como gradientes viram aprendizado? Gradiente descendente do zero”**. Os resultados são medidos; nenhum script altera métricas ou força divergência para sustentar a narrativa.

## Preparação

Requer Python 3.11+ e [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

`uv sync` cria `.venv` a partir de `pyproject.toml` e `uv.lock`. O MNIST é baixado por `torchvision` somente quando um dos scripts 04–06 é executado. Matplotlib usa o backend headless `Agg`.

## Ordem de execução

Cada script aceita `--seed` e `--output-dir`; use `--help` para todas as opções. Os defaults usam seed 42.

```bash
# 1) Uma regressão do início ao fim, com dw e db explícitos
uv run python scripts/01_regressao_gd_manual.py

# 2) Mesmo dataset e mesmos w,b iniciais, três learning rates
uv run python scripts/02_regressao_comparar_learning_rates.py

# 3) w,b congelados: dispersão de muitos gradientes por batch size
uv run python scripts/03_ruido_gradiente_por_batch_size.py

# 4) MLP + mini-batch + CrossEntropyLoss + torch.optim.SGD
uv run python scripts/04_mnist_treino_sgd.py

# 5) Mesma inicialização e ordem de mini-batches, três learning rates
uv run python scripts/05_mnist_comparar_learning_rates.py

# 6) Grade visual a partir do checkpoint produzido pelo script 04
uv run python scripts/06_mnist_grade_previsoes.py
```

Para uma passagem rápida pelo MNIST (download ainda necessário):

```bash
uv run python scripts/04_mnist_treino_sgd.py --quick
uv run python scripts/05_mnist_comparar_learning_rates.py --quick
uv run python scripts/06_mnist_grade_previsoes.py --count 8
```

## Reprodutibilidade e interpretação

- A regressão usa `numpy.random.default_rng(seed)` e registra inclusive o estado no passo zero.
- No experimento 03, **não há atualização**: todos os gradientes são avaliados no mesmo `w,b`. A estrela no plot é o gradiente full batch.
- Na comparação MNIST, o `state_dict` inicial é clonado e o `DataLoader` é recriado com a mesma seed para cada learning rate.
- A acurácia MNIST é calculada no conjunto oficial de teste, separado do treino.
- A curva pode não divergir com o maior learning rate: relate o que os CSVs mostram. Troque a lista com `--learning-rates`, sem ocultar resultados.
- `data/` e `outputs/` são locais e ignorados pelo Git; os arquivos `.gitkeep` preservam somente as pastas.

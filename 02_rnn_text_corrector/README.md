# 02_rnn_text_corrector

<a href="https://youtube.com/playlist?list=PLxn5KTcVccYFDGqKMaqfa62RFj2Q5WAaN&si=j8Mnj_LdGv3jENCH" target="_blank">
  <img src="https://img.shields.io/badge/Assistir_no_YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir no YouTube"/>
</a>

## 📝 Sobre este projeto
Neste projeto implementaremos uma rede neural recorrente (RNN) no formato encoder-decoder que fará correção de texto!

**Tópicos abordados:**
- Dados e preparação - [1_DATAPREP.ipynb](./1_DATAPREP.ipynb)
- Remoção de duplicidade de textos com Hash e MinHashLSH ()
    - Remoção de duplicidade padrão em memória RAM - [duplicidade.py](./duplicidade.py)
    - Remoção de duplicidade customizada para caber na memória - [duplicidade_custom.py](./duplicidade_custom.py)
    - Remoção de duplicidade customizada e RÁPIDA - [duplicidade_custom_parallel.py](./duplicidade_custom_parallel.py)
- Criação dos componentes (tokenizador, dataset, scheduler e modelo) - [src](./src/)
- Loop de treinamento [train.py](train.py) e Loop de treinamento avançado [train_plus.py](train_plus.py)
- (INCOMPLETO) Script de inferência [inference.py](inference.py)
- (EM BREVE) Script de avaliação

**Datasets mostrados no vídeo**:
* [Wikipedia PT Dump](https://dumps.wikimedia.org/ptwiki/20260101/)
* [OPUS - OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles/en&pt/v2024/OpenSubtitles)

**Datasets que eu utilizei**:
- [Base do wikipedia (202512) tratada](https://drive.google.com/file/d/1JGthoy7aWbU9xz1rGoRxD_epaRZGAsSI/view?usp=sharing)
- [Dataset de treinamento (sem duplicidade)](https://drive.google.com/file/d/1pqgHJd-VplJOabLgvcHv93quAO7Uv6KB/view?usp=sharing)
- [Dataset de validação](https://drive.google.com/file/d/1PlQJbCxcrCQFFKyRUziaWVQTSyGcDWEr/view?usp=sharing)
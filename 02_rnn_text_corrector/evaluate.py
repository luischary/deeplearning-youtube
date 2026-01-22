import re
import json

import torch
import pandas as pd

from inference import (
    generate_with_attention,
    generate,
    beam_search_decode,
    inference_with_attention,
)
from src.tokenizer import Tokenizer
from src.model import AttentionCorrector, Corretor

pd.set_option("display.max_rows", None)
device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúâêôãõç]", "", text)
    return text.strip()


def wer_metric(gen: str, ref: str) -> float:
    """
    Calcula o WER (word error rate) da string gerada com base na referencia
    """

    gen_words = normalize_for_wer(gen).split()
    ref_words = normalize_for_wer(ref).split()

    if len(ref_words) == 0:
        return 0.0 if len(gen_words) == 0 else 1.0

    # matriz dp
    dp = [[0] * (len(gen_words) + 1) for _ in range(len(ref_words) + 1)]

    # inicializacao
    for i in range(len(ref_words) + 1):
        dp[i][0] = i  # delecoes
    for j in range(len(gen_words) + 1):
        dp[0][j] = j  # insercoes

    # preenchimento
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(gen_words) + 1):
            if ref_words[i - 1] == gen_words[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delecao
                dp[i][j - 1] + 1,  # insercao
                dp[i - 1][j - 1] + cost,  # substituicao
            )

    return dp[-1][-1] / len(ref_words)


def normalize_for_cer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cer_metric(gen: str, ref: str) -> float:
    """
    Calcula o CER (character error rate) da string gerada com base na referencia
    """

    gen_words = list(normalize_for_cer(gen))
    ref_words = list(normalize_for_cer(ref))

    if len(ref_words) == 0:
        return 0.0 if len(gen_words) == 0 else 1.0

    # matriz dp
    dp = [[0] * (len(gen_words) + 1) for _ in range(len(ref_words) + 1)]

    # inicializacao
    for i in range(len(ref_words) + 1):
        dp[i][0] = i  # delecoes
    for j in range(len(gen_words) + 1):
        dp[0][j] = j  # insercoes

    # preenchimento
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(gen_words) + 1):
            if ref_words[i - 1] == gen_words[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delecao
                dp[i][j - 1] + 1,  # insercao
                dp[i - 1][j - 1] + cost,  # substituicao
            )

    return dp[-1][-1] / len(ref_words)


def eval_(eval_set, modelo, tokenizer, name: str, gen_function):
    eval_df = eval_set.copy()
    eval_df["generated"] = eval_df["src"].apply(
        lambda x: gen_function(modelo, x, tokenizer, max_len=200)
    )
    eval_df["wer"] = eval_df.apply(
        lambda row: wer_metric(row["generated"], row["tgt"]), axis=1
    )
    eval_df["cer"] = eval_df.apply(
        lambda row: cer_metric(row["generated"], row["tgt"]), axis=1
    )
    print(eval_df)
    resultado = eval_df.groupby("cat").agg({"wer": "mean", "cer": "mean"})
    resultado = resultado.reset_index().rename(
        columns={"wer": f"wer_{name}", "cer": f"cer_{name}"}
    )
    resultado = pd.concat(
        [
            resultado,
            pd.DataFrame(
                {
                    "cat": ["GERAL"],
                    f"wer_{name}": [eval_df["wer"].mean()],
                    f"cer_{name}": [eval_df["cer"].mean()],
                }
            ),
        ],
        ignore_index=True,
    )
    return resultado


def eval_model(eval_set, model, tokenizer, name, gen_function):
    print("Avaliando modelo")
    resultado = eval_(eval_set, model, tokenizer, name, gen_function)
    return resultado


if __name__ == "__main__":
    dados = json.load(open("data/eval_set.json", "r", encoding="utf8"))
    eval_set = (
        pd.DataFrame(dados)
        .sort_values(by="cat")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(eval_set)

    resultados = []

    tokenizer = Tokenizer()

    # modelo = Corretor(
    #     vocab_size=100 + 4,
    #     embed_dim=256,
    #     hidden_encoder=512,
    #     hidden_decoder=512,
    #     num_layers_encoder=2,
    #     num_layers_decoder=2,
    # )
    # modelo.load_state_dict(
    #     torch.load(
    #         "modelos_treinados/apelao_256_512_2l_noatt/last.pt", map_location="cpu"
    #     )
    # )

    modelo = AttentionCorrector(
        vocab_size=100 + 4,
        embed_dim=256,
        hidden_encoder=512,
        hidden_decoder=512,
        num_layers_encoder=2,
        num_layers_decoder=2,
    )
    modelo.load_state_dict(
        torch.load("modelos_treinados/apelao_256_512_2l_ultimate/last.pt")
    )
    modelo.to(device)
    modelo.eval()

    resultado = eval_model(eval_set, modelo, tokenizer, "inf", inference_with_attention)
    print(resultado)

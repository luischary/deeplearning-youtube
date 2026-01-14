import re


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

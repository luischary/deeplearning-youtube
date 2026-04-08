from pathlib import Path
import os
import time

import torch
import numpy as np

from src.model import DecoderLM
from src.tokenizer import BPETokenizer, SPECIAL_TOKENS
from utils import load_torch_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
# controla aleatoriedade da geração
SEED = 41
torch.manual_seed(SEED)
np.random.seed(SEED)


# greedy decoding
def generate(model, prompt, tokenizer, max_len: int = 200):
    tokens = tokenizer.encode(prompt)
    tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens

    for _ in range(len(tokens), max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.int).to(device)
        with torch.no_grad():
            logits = model(input_tokens)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).detach().cpu().item()
        tokens.append(next_token)

        if next_token == SPECIAL_TOKENS["<EOS>"]:
            break

    return tokenizer.decode(tokens)


def generate_by_sample(
    model, prompt, tokenizer, temperature: float, max_len: int = 200
):
    tokens = tokenizer.encode(prompt)
    tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens

    for _ in range(len(tokens), max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.int).to(device)
        with torch.no_grad():
            logits = model(input_tokens)
        next_token_logits = logits[0, -1, :] / temperature
        probas = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probas, num_samples=1).detach().cpu().item()
        tokens.append(next_token)

        if next_token == SPECIAL_TOKENS["<EOS>"]:
            break

    return tokenizer.decode(tokens)


def generate_by_nucleus(
    model,
    prompt,
    tokenizer,
    top_p: float = 0.9,
    temperature: float = 1.0,
    max_len: int = 200,
):
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p deve estar no intervalo (0, 1].")

    if temperature <= 0:
        raise ValueError("temperature deve ser > 0.")

    tokens = tokenizer.encode(prompt)
    tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens

    for _ in range(len(tokens), max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tokens)

        next_token_logits = logits[:, -1, :] / temperature
        probas = torch.softmax(next_token_logits, dim=-1)

        # Ordena probabilidades em ordem decrescente
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

        # Soma acumulada
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mantém apenas o menor conjunto com soma acumulada <= top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Garante que pelo menos o token mais provável esteja disponível
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[:, indices_to_remove] = -float("Inf")

        # Amostra dentro do núcleo
        next_token_id = torch.multinomial(
            F.softmax(next_token_logits, dim=-1), num_samples=1
        ).item()

        tokens.append(next_token_id)

        if next_token_id == SPECIAL_TOKENS["<EOS>"]:
            break

    return tokenizer.decode(tokens), len(tokens)


def generate_with_effect(
    model,
    prompt,
    tokenizer,
    top_p: float = 0.9,
    temperature: float = 1.0,
    max_len: int = 200,
):
    gerado = generate_by_nucleus(
        model, prompt, tokenizer, top_p=top_p, temperature=temperature
    )
    for i in range(len(prompt), len(gerado)):
        print("\r" + gerado[:i], end="")
        time.sleep(0.05)
    print()  # para garantir que o cursor vá para a próxima linha


if __name__ == "__main__":
    MAX_LEN = 512

    tokenizer = BPETokenizer(
        merges_path="artifacts/bpe_tokenizer_100k.json", vocab_size=10_000
    )

    # modelo = DecoderLM(
    #     vocab_size=10_000,
    #     embed_dim=256,
    #     num_heads=8,
    #     hidden_size=1024,
    #     num_layers=4,
    #     max_len=MAX_LEN,
    #     dropout=0.1,
    # )
    # modelo.load_state_dict(
    #     torch.load("modelos_treinados/teste_decoder_wiki/last.pt", map_location="cpu")
    # )

    modelo = DecoderLM(
        vocab_size=10_000,
        embed_dim=768,
        num_heads=12,
        hidden_size=768 * 4,
        num_layers=8,
        max_len=MAX_LEN,
        dropout=0.1,
    )
    modelo.load_state_dict(
        load_torch_checkpoint(
            "modelos_treinados/decoder_wiki_base/last.pt",
            map_location="cpu",
            weights_only=False,
        )["modelo"]
    )

    modelo.to(device)
    modelo.eval()

    tic = time.time()
    total_tokens = 0
    for prompt in [
        "O filme foi ",
        "O produto chegou ",
        "O Brasil é ",
        "Era uma noite escura e ",
        "Explique o que é uma rede neural",
        "Qual é a capital do Brasil?",
        # "O roteiro é muito bom e o roteiro é muito",
        # "Este filme foi a pior experiência da minha vida. A unica surpresa do filme é que ",
        # "Batman decide salvar a cidade usando apenas o seu ",
        # "Comprei este smartphone para o meu gato ",
        # "Uma comédia romântica sobre um tubarão que ",
        # "Luis Felipe Chary é um pesquisador conhecido ",
        # "Inteligência artificial é uma área fascinante que tem o potencial de ",
        # "As gírias mais comuns hoje em dia e seus significados são:",
        # "Sinopse\nA história de terror começa com um peixe atirador ",
        # "Matriculei o meu cachorro na escola de Jedi para ele aprender a usar a Força. ",
    ]:
        # gerado = generate(modelo, prompt, tokenizer, MAX_LEN)
        # print(gerado)
        # generate_with_effect(
        #     modelo, prompt, tokenizer, top_p=0.9, temperature=0.6, max_len=MAX_LEN
        # )

        # gerado = generate(modelo, prompt, tokenizer, MAX_LEN)
        # print(gerado)
        # gerado = generate_by_sample(
        #     modelo, prompt, tokenizer, temperature=0.7, max_len=MAX_LEN
        # )
        # print(gerado)
        # gerado = generate_by_nucleus(
        #     modelo, prompt, tokenizer, top_p=0.9, temperature=0.7, max_len=MAX_LEN
        # )
        # print(gerado)

        for t in range(6, 9):
            temp = t / 10.0
            for _ in range(1):
                # gerado = generate_by_sample(
                #     modelo, prompt, tokenizer, temperature=temp, max_len=MAX_LEN
                # )
                gerado, num_tokens = generate_by_nucleus(
                    modelo,
                    prompt,
                    tokenizer,
                    top_p=0.9,
                    temperature=temp,
                    max_len=MAX_LEN,
                )
                total_tokens += num_tokens
                print(temp, "-", gerado)
        print("#" * 50)
    tac = time.time()
    print(f"Tempo total: {tac - tic:.2f} segundos para gerar {total_tokens} tokens")
    print(f"{total_tokens / (tac - tic):.2f} tokens por segundo")

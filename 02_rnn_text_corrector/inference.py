import re

import torch
import torch.nn.functional as F

from src.model import Corretor, AttentionCorrector
from src.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

from spellchecker import SpellChecker

device = "cuda" if torch.cuda.is_available() else "cpu"
spell = SpellChecker(language="pt")


def arruma_tamanho(tokens, max_len: int):
    tamanho = len(tokens)
    if tamanho < max_len:
        tokens += [PAD_TOKEN] * (max_len - tamanho)
    else:
        tokens = tokens[:max_len]
    return tokens


def generate(model, ref_text, tokenizer, max_len):
    tokens_ref = [BOS_TOKEN] + tokenizer.tokenize(ref_text) + [EOS_TOKEN]
    tokens_ref = arruma_tamanho(tokens_ref, max_len)
    input_ref = torch.tensor([tokens_ref], dtype=torch.int, device=device)

    tokens_gen = [BOS_TOKEN]

    # roda encoder
    with torch.no_grad():
        _, (h, c) = model.encoder(input_ref)
        h, c = model.project_memory(h, c)

    for _ in range(max_len):
        input_tokens = torch.tensor([[tokens_gen[-1]]], dtype=torch.int, device=device)
        with torch.no_grad():
            logits, (h, c) = model.decoder(input_tokens, h, c)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits).detach().cpu().item()
        tokens_gen.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokenizer.detokenize(tokens_gen)


def generate_with_attention(model, ref_text, tokenizer, max_len):
    tokens_ref = [BOS_TOKEN] + tokenizer.tokenize(ref_text) + [EOS_TOKEN]
    tokens_ref = arruma_tamanho(tokens_ref, max_len)
    input_ref = torch.tensor([tokens_ref], dtype=torch.int, device=device)

    tokens_gen = [BOS_TOKEN]

    # roda encoder
    with torch.no_grad():
        encoded, _ = model.encoder(input_ref)
        encoded = model.proj_encoded(encoded)
        encoded = model.norm_encoded(encoded)

    h, c = None, None
    for _ in range(max_len):
        input_tokens = torch.tensor([[tokens_gen[-1]]], dtype=torch.int, device=device)
        with torch.no_grad():
            logits, (h, c) = model.decoder(encoded, input_tokens, h, c)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits).detach().cpu().item()
        tokens_gen.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokenizer.detokenize(tokens_gen)


def spell_rerank(candidates, penalty_per_error=3.0):
    """
    Recebe uma lista de tuplas (score, texto) e reordena pelos erros de ortografia
    """
    reranked = []
    for score, texto in candidates:
        words = (
            texto.replace("?", "")
            .replace("!", "")
            .replace(".", "")
            .replace(":", "")
            .split()
        )
        desconhecidos = spell.unknown(words)
        penalty = len(desconhecidos) * penalty_per_error
        reranked.append((score - penalty, texto))

    # reordena pelo novo score
    reranked = sorted(reranked, key=lambda x: x[0], reverse=True)
    return reranked


def beam_search_decode(
    model,
    ref_text,
    tokenizer,
    beam_width: int = 3,
    max_len: int = 200,
    spellcheck: bool = True,
):
    tokens_ref = [BOS_TOKEN] + tokenizer.tokenize(ref_text) + [EOS_TOKEN]
    tokens_ref = arruma_tamanho(tokens_ref, 200)
    input_ref = torch.tensor([tokens_ref], dtype=torch.int).to(device)

    # roda o encoder 1 vez so
    with torch.no_grad():
        encoded, _ = model.encoder(input_ref)
        encoded = model.proj_encoded(encoded)
        encoded = model.norm_encoded(encoded)

    h, c = None, None
    # estrutura do beam: (score_acumulado, tokens_gerados, (h, c))
    k_beams = [(0.0, [BOS_TOKEN], (h, c))]

    for _ in range(max_len):
        candidates = []

        # para cada caminho (beam) atual
        for score, tokens_seq, (h, c) in k_beams:
            if tokens_seq[-1] == EOS_TOKEN:
                # se já terminou, mantém o beam
                candidates.append((score, tokens_seq, (h, c)))
                continue

            # roda o decoder para o ultimo tokens gerado
            dec_input = torch.tensor([[tokens_seq[-1]]], dtype=torch.int).to(device)
            with torch.no_grad():
                # logits, (h_new, c_new) = model(input_ref, dec_input, h, c)
                logits, (h_new, c_new) = model.decoder(encoded, dec_input, h, c)

            # logprob
            log_prob = F.log_softmax(logits[0, -1, :], dim=-1)

            # pega os top K deste passo
            topk_logprobs, topk_indices = torch.topk(log_prob, beam_width)

            for k in range(beam_width):
                idx = topk_indices[k].detach().cpu().item()
                prob = topk_logprobs[k].detach().cpu().item()

                new_seq = tokens_seq + [idx]
                new_score = score + prob
                candidates.append((new_score, new_seq, (h_new, c_new)))

        # ordena pelo score e fica com os top K
        k_beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # se todos os beams terminarem com EOS, para
        if all(beam[1][-1] == EOS_TOKEN for beam in k_beams):
            break

    if spellcheck:
        # faz o reranking pelo corretor ortografico
        candidatos = []
        for score, tokens_seq, (h, c) in k_beams:
            texto = tokenizer.detokenize(tokens_seq)
            candidatos.append((score, texto))
        ordenados = spell_rerank(candidatos, penalty_per_error=3.0)
        # for score, texto in ordenados:
        #     print(f"Score: {score} - Seq: {texto}")
        # vencedor
        best_seq = ordenados[0][1]
        return best_seq

    # retorna a melhor sequencia
    best_seq = k_beams[0][1]

    # for score, tokens_seq, (h, c) in k_beams:
    #     print(f"Score: {score} - Seq: {tokenizer.detokenize(tokens_seq)}")
    return tokenizer.detokenize(best_seq)


# para quebrar as frases
def split_sentences(texto):
    sentences = re.findall(r".+?[.!?]+(?=\s+|$)", texto)
    if len(sentences) == 0:
        sentences = [texto]

    # pode ser que alguma frase nao termine em pontuacao, entao...
    consumido = " ".join(sentences)
    sobra = texto[len(consumido) :].strip()
    if sobra:
        sentences.append(sobra)
    return [s.strip() for s in sentences]


def inference_with_attention(
    model,
    ref_text,
    tokenizer,
    beam_width: int = 3,
    max_len: int = 200,
    spellcheck=True,
):

    novas = []
    sentencas = split_sentences(ref_text)
    for sent in sentencas:
        correcao = beam_search_decode(
            model,
            sent,
            tokenizer,
            beam_width=beam_width,
            max_len=max_len,
            spellcheck=spellcheck,
        )
        novas.append(correcao)
    return " ".join(novas)


if __name__ == "__main__":
    MAX_LEN = 100
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

    for texto in [
        "este aqui é u testw para vodê ver a coreçãaaao",
        "estou bem e vc?",
        "tom com fomeeee, quero pizaa!!",
        "Mds, o q vc ta fznd ai???",
        "eh isto, nao tem o q fazer, aceita q doi menos",
        "digitnaod super rapido aqui no meu teclado",
    ]:
        # print(generate(modelo, texto, tokenizer, MAX_LEN))
        print(generate_with_attention(modelo, texto, tokenizer, MAX_LEN))

import torch
import torch.nn.functional as F

from src.model import Corretor, AttentionCorrector
from src.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    # modelo.load_state_dict(torch.load("modelo_treinado_big.pt", map_location="cpu"))

    modelo = AttentionCorrector(
        vocab_size=100 + 4,
        embed_dim=128,
        hidden_encoder=256,
        hidden_decoder=256,
        num_layers_encoder=1,
        num_layers_decoder=1,
    )
    modelo.load_state_dict(
        torch.load("modelos_treinados/pequeno_128_256_1l/modelo_treinado.pt")
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

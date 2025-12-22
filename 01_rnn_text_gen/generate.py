import torch

from src.model import Modelo
from src.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate(model, prompt, tokenizer, max_len: int = 200):
    tokens = tokenizer.tokenize(prompt)
    tokens = [BOS_TOKEN] + tokens

    h, c = None, None
    for _ in range(max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.int).to(device)
        with torch.no_grad():
            logits, (h, c) = model(input_tokens, h, c)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).detach().cpu().item()
        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokenizer.detokenize(tokens)


def generate_by_sample(
    model, prompt, tokenizer, temperature: float, max_len: int = 200
):
    tokens = tokenizer.tokenize(prompt)
    tokens = [BOS_TOKEN] + tokens

    h, c = None, None
    for _ in range(max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.int).to(device)
        with torch.no_grad():
            logits, (h, c) = model(input_tokens, h, c)
        next_token_logits = logits[0, -1, :] / temperature
        probas = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probas, num_samples=1).detach().cpu().item()
        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokenizer.detokenize(tokens)


if __name__ == "__main__":
    tokenizer = Tokenizer()

    modelo = Modelo(vocab_size=1409 + 3, embed_dim=256, hidden_size=512, n_layers=2)
    modelo.load_state_dict(torch.load("modelo_treinado_big.pt", map_location="cpu"))
    modelo.to(device)
    modelo.eval()

    for prompt in [
        # "O roteiro é muito bom e o roteiro é muito",
        # "Este filme foi a pior experiência da minha vida, mas adorei",
        "Batman decide salvar a cidade usando apenas o seu",
        "Comprei este smartphone para o meu gato e",
        "Uma comédia romântica sobre um tubarão que",
    ]:
        for t in range(1, 6):
            temp = t / 10.0
            for _ in range(2):
                gerado = generate_by_sample(modelo, prompt, tokenizer, temperature=temp)
                print(temp, "-", gerado)

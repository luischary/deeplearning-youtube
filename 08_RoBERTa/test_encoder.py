import torch

from src.model import EncoderMLM
from src.tokenizer import BPETokenizer, SPECIAL_TOKENS

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample(model, text, tokenizer):
    tokens = tokenizer.encode(prompt)
    tokens = [SPECIAL_TOKENS["<BOS>"]] + tokens + [SPECIAL_TOKENS["<EOS>"]]

    mask = [i == SPECIAL_TOKENS["<MASK>"] for i in tokens]

    # mask = torch.tensor(mask, dtype=torch.bool)
    tokens_t = torch.tensor([tokens], dtype=torch.int).to(device)
    with torch.no_grad():
        logits = model(tokens_t).detach().squeeze().cpu()
    for i in range(len(mask)):
        if mask[i]:
            probas = torch.softmax(logits[i], dim=-1)
            ordenado, indices = torch.sort(probas, descending=True)

            for k in range(3):
                print(
                    f"{i + 1} - {ordenado[k]:.4f} - {tokenizer.decode([indices[k].item()])}"
                )
            # for k in range(3):
            #     recriado = []
            #     for j in range(len(tokens)):
            #         if mask[j]:
            #             recriado.append(indices[k].item())
            #         else:
            #             recriado.append(tokens[j])
            #     texto_recriado = tokenizer.decode(recriado)
            #     print(f"{ordenado[k]:.4f} - {texto_recriado}")


if __name__ == "__main__":
    MAX_LEN = 80

    tokenizer = BPETokenizer(
        merges_path="artifacts/bpe_tokenizer_100k.json", vocab_size=10_000
    )

    modelo = EncoderMLM(
        vocab_size=10_000,
        embed_dim=256,
        num_heads=8,
        hidden_size=1024,
        num_layers=4,
        max_len=MAX_LEN,
        dropout=0.1,
    )
    modelo.load_state_dict(
        torch.load(
            "modelos_treinados/teste_encoder/modelo_treinado.pt", map_location="cpu"
        )
    )
    modelo.to(device)
    modelo.eval()

    for prompt in [
        "O roteiro é muito bom e o roteiro é <MASK>",
        "Este filme foi o pior da minha vida. Eu <MASK>gostei.",
        "Eu achei legal. O final foi <MASK>.",
        "The cat is <MASK>on the table.",
    ]:
        print(prompt)
        gerado = sample(modelo, prompt, tokenizer)
        print()

    tokenizer = BPETokenizer(
        merges_path="artifacts/tokenizer_wiki_en_5k.json", vocab_size=5_000
    )

    modelo = EncoderMLM(
        vocab_size=5_000,
        embed_dim=256,
        num_heads=8,
        hidden_size=1024,
        num_layers=4,
        max_len=MAX_LEN,
        dropout=0.1,
    )
    modelo.load_state_dict(
        torch.load(
            "modelos_treinados/encoder_mlm_wiki_en_5k_200k/last.pt", map_location="cpu"
        )
    )
    modelo.to(device)
    modelo.eval()

    for prompt in [
        "The cat is <MASK>on the table.",
    ]:
        print(prompt)
        gerado = sample(modelo, prompt, tokenizer)
        print()

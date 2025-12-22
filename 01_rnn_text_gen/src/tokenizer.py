import json
from pathlib import Path

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2


class Tokenizer:
    def __init__(
        self,
        tokenize_dict_path: str = "artifacts/tokenize_dict.json",
        detokenize_dict_path: str = "artifacts/detokenize_dict.json",
    ):

        self.tokenize_dict = json.loads(
            Path(tokenize_dict_path).read_text(encoding="utf8")
        )
        self.detokenize_dict = json.loads(
            Path(detokenize_dict_path).read_text(encoding="utf8")
        )

        self.tokenize_dict = {
            chave: valor + 3 for chave, valor in self.tokenize_dict.items()
        }
        self.detokenize_dict = {
            int(chave) + 3: valor for chave, valor in self.detokenize_dict.items()
        }

    def tokenize(self, texto):
        tokens = []
        for char in texto:
            tokens.append(self.tokenize_dict[char])
        return tokens

    def detokenize(self, tokens):
        texto = ""
        for token in tokens:
            if token not in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]:
                texto += self.detokenize_dict[token]
        return texto


if __name__ == "__main__":
    tokenizer = Tokenizer(
        tokenize_dict_path="./../artifacts/tokenize_dict.json",
        detokenize_dict_path="./../artifacts/detokenize_dict.json",
    )

    frase = "Testando 1, 2, 3.... ALO!"
    print(frase)
    tokens = tokenizer.tokenize(frase)
    print(tokens)
    detokenized = tokenizer.detokenize(tokens)
    print(detokenized)

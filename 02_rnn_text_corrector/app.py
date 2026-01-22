import difflib

import torch
from flask import Flask, render_template, request, jsonify

from src.tokenizer import Tokenizer
from src.model import AttentionCorrector
from inference import inference_with_attention

app = Flask(__name__)

## load no modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo = AttentionCorrector(
    vocab_size=100 + 4,
    embed_dim=256,
    hidden_encoder=512,
    hidden_decoder=512,
    num_layers_encoder=2,
    num_layers_decoder=2,
)
modelo.load_state_dict(
    torch.load("modelos_treinados/apelao_256_512_2l_ultimate_200/last.pt")
)
modelo.to(device)
modelo.eval()

# tokenizer
tokenizer = Tokenizer()


def gerar_diff_visual(original, corrigido):
    s = difflib.SequenceMatcher(None, original, corrigido)
    html_output = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            html_output.append(original[i1:i2])
        elif tag in ["replace", "delete"]:
            # Envolve o erro no span com a ondinha vermelha
            html_output.append(f'<span class="erro">{original[i1:i2]}</span>')
        # Inserções ignoramos no diff visual do original
    return "".join(html_output)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    texto_raw = data.get("text", "")

    texto_corrigido = inference_with_attention(
        modelo, texto_raw, tokenizer, beam_width=5, max_len=200, spellcheck=False
    )

    return jsonify(
        {
            "html_diff": gerar_diff_visual(texto_raw, texto_corrigido),
            "corrected_plain": texto_corrigido,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

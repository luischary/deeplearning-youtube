import re
import random

from unidecode import unidecode

SEPARA_PALAVRAS = re.compile(r"\w+|\S", re.UNICODE)


def separate_words(text):
    return SEPARA_PALAVRAS.findall(text)


def join_words(palavras):
    out = []
    for i, t in enumerate(palavras):
        if i == 0:
            out.append(t)
            continue
        # se a palavra atual e uma pontuacao junta, nao coloca espaco
        if re.match(r"^[\.\,\!\?\:\;\)\]]+$", t):
            out.append(t)
        # se palavra anterior e uma abertura, nao coloca espaco
        elif re.match(r"^[\(\[\{]+$", palavras[i - 1]):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)


# INFORMAL
def default_informal_map():
    # chaves devem estar em lower()
    # valores: lista de possíveis substituições
    return {
        "você": ["vc"],
        "vocês": ["vcs"],
        "porque": ["pq"],
        "por que": ["pq"],
        "porquê": ["pq"],
        "não": ["nao", "n"],
        "também": ["tb", "tambem"],
        "para": ["pra"],
        "para o": ["pro"],
        "está": ["ta", "tá"],
        "estou": ["to", "tô"],
        "estava": ["tava"],
        "estavam": ["tavam"],
        "estamos": ["tamo", "tamos"],
        "estão": ["tao", "estao"],
        "beleza": ["blz"],
        "quem": ["qm"],
        "que": ["q"],
        "com": ["c"],
        "sem": ["s"],
        "de": ["d"],
        "meu": ["meu"],
        "minha": ["minha"],
        "e": ["e"],
        "é": ["e", "eh"],
        "ok": ["blz", "ok"],
        "obrigado": ["vlw", "obg", "valeu"],
        "obrigada": ["vlw", "obg", "valeu"],
        "hoje": ["hj"],
        "amanhã": ["amanha", "amnh", "amanh"],
        "tudo": ["td"],
        "mensagem": ["msg"],
        "mesmo": ["msm"],
        "por favor": ["pfv"],
        "depois": ["dps"],
        "o que": ["oq"],
        "demais": ["dms"],
        "muita": ["mta"],
        "muito": ["mto"],
        "meu deus": ["mds"],
    }


def informalize_words(palavras, proba: float):
    new_words = []
    informal_map = default_informal_map()
    pula = False

    for i in range(len(palavras)):
        p = palavras[i]
        if i < len(palavras) - 1:
            proxima = palavras[i + 1]
        else:
            proxima = None

        if pula:
            pula = False
            continue

        # bigrama
        if proxima is not None:
            bigrama = " ".join([p.lower(), proxima.lower()])
            if bigrama in informal_map:
                if random.random() <= proba:
                    new_words.append(random.choice(informal_map[bigrama]))
                    pula = True
                    continue

        # unigrama
        if p.lower() in informal_map:
            if random.random() <= proba:
                new_words.append(random.choice(informal_map[p.lower()]))
                continue

        new_words.append(p)
    return new_words


# ERROS DE DIGITACAO
# imita erros de teclas vizinhas no teclado
QUERTY_NEIGHBORS = {
    "a": "sqz",
    "b": "vgn",
    "c": "xdfvk",
    "d": "sefxc",
    "e": "wrdi",
    "f": "drgcv",
    "g": "fvtbh",
    "h": "gybnj",
    "i": "ujkoe",
    "j": "hkiunm",
    "k": "lmjopc",
    "l": "kop",
    "m": "njkl",
    "n": "bhmj",
    "o": "iklp",
    "p": "ol",
    "q": "aswk",
    "r": "edftg",
    "s": "azxdew",
    "t": "rfghy",
    "u": "yjhki",
    "v": "cbfg",
    "w": "qasde",
    "x": "zsdc",
    "y": "tghju",
    "z": "asx",
}


def keyboard_error(texto, proba: float):
    new_text = ""
    for char in texto:
        if char.lower() not in QUERTY_NEIGHBORS:
            new_text += char
            continue

        if random.random() <= proba:
            novo = random.choice(QUERTY_NEIGHBORS[char.lower()])
            if char.isupper():
                novo = novo.upper()
            new_text += novo
        else:
            new_text += char
    return new_text


# REPETICAO
def repeat_char(texto, proba: float):
    novo_texto = ""
    for char in texto:
        novo_texto += char
        contagem = 0
        while random.random() <= proba and contagem <= 5:
            novo_texto += char
            contagem += 1
    return novo_texto


# SWAP DE CHARACTERES
def swap_chars(texto, proba: float):
    chars = list(texto)
    i = 0
    while i < len(chars) - 1:
        if random.random() <= proba:
            # troca
            temp = chars[i]
            chars[i] = chars[i + 1]
            chars[i + 1] = temp
            i += 2  # pula o proximo
        else:
            i += 1
    return "".join(chars)


# REMOCAO
def remove_char(texto, proba: float):
    novo_texto = ""
    for char in texto:
        if random.random() <= proba:
            continue
        novo_texto += char
    return novo_texto


# UPPER LOWER E ACENTOS
def all_upper(texto, proba: float):
    if random.random() <= proba:
        return texto.upper()
    return texto


def all_lower(texto, proba: float):
    if random.random() <= proba:
        return texto.lower()
    return texto


def remove_acentos(texto, proba: float):
    if random.random() <= proba:
        return unidecode(texto)
    return texto


def text_augmentation(
    texto: str,
    proba_informalize: float = 0.5,
    proba_remove_acentos: float = 0.3,
    proba_swap_chars: float = 0.05,
    proba_keyboard_error: float = 0.05,
    proba_all_upper: float = 0.05,
    proba_all_lower: float = 0.05,
    proba_repeat_char: float = 0.02,
    proba_remove_char: float = 0.01,
):
    # 15% de chance de nao mudar nada
    if random.random() <= 0.15:
        return texto

    words = separate_words(texto)
    texto = join_words(informalize_words(words, proba_informalize))

    texto = remove_acentos(texto, proba_remove_acentos)
    texto = swap_chars(texto, proba_swap_chars)
    texto = keyboard_error(texto, proba_keyboard_error)
    texto = all_upper(texto, proba_all_upper)
    texto = all_lower(texto, proba_all_lower)
    texto = repeat_char(texto, proba_repeat_char)
    texto = remove_char(texto, proba_remove_char)

    return texto


if __name__ == "__main__":
    frase = "Olá, tudo bem com você? Espero que sim!"
    print("Original: ", frase)
    for _ in range(10):
        augmentada = text_augmentation(frase)
        print("Augmentada:", augmentada)

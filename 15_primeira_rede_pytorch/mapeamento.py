"""Gera metadados path/label para os splits train e test do MNIST exportado."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def mapear_split(split: str) -> None:
    images_root = DATA_DIR / "MNIST" / "images" / split
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Imagens ausentes em {images_root}. Execute antes "
            "data/download_mnist.py e data/prepare_mnist.py."
        )

    dados = {"path": [], "label": []}
    for pasta_classe in sorted(images_root.iterdir()):
        if not pasta_classe.is_dir():
            continue
        for image_path in sorted(pasta_classe.glob("*.png")):
            dados["path"].append("./" + image_path.relative_to(DATA_DIR.parent).as_posix())
            dados["label"].append(int(pasta_classe.name))

    df = pd.DataFrame(dados)
    output = DATA_DIR / f"metadados_{'treino' if split == 'train' else 'teste'}.csv"
    df.to_csv(output, index=False)
    print(f"{split}: {len(df)} exemplos -> {output}")


if __name__ == "__main__":
    for split in ("train", "test"):
        mapear_split(split)

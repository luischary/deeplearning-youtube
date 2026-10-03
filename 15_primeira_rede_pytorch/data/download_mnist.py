"""Baixa e extrai os quatro arquivos IDX originais do MNIST via torchvision."""

from pathlib import Path

from torchvision.datasets import MNIST


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    MNIST(root=data_dir, train=True, download=True)

    raw_dir = data_dir / "MNIST" / "raw"
    expected = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    missing = [name for name in expected if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos IDX ausentes em {raw_dir}: {missing}")
    print(f"Arquivos IDX prontos em {raw_dir}")


if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image

from original_loader import MnistDataloader

(x_train, y_train), (x_test, y_test) = MnistDataloader(
    "./MNIST/raw/train-images-idx3-ubyte",
    "./MNIST/raw/train-labels-idx1-ubyte",
    "./MNIST/raw/t10k-images-idx3-ubyte",
    "./MNIST/raw/t10k-labels-idx1-ubyte",
).load_data()


def export_mnist_to_folders(images, labels, output_dir):
    """Salva imagens em subpastas nomeadas pelo rótulo:

    output_dir/
      ├── 0/
      ├── 1/
      ...
      └── 9/
    """
    # Cria as pastas de 0 a 9 antecipadamente
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    total = len(images)
    print(f"Exportando {total} imagens para '{output_dir}'...")

    for idx, (img_data, label) in enumerate(zip(images, labels)):
        # Garante o formato uint8 (28, 28)
        img_array = np.array(img_data, dtype=np.uint8)

        # Converte o array para objeto de imagem e salva em PNG (sem perdas)
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, str(label), f"img_{idx:05d}.png")
        img.save(img_path)

        if (idx + 1) % 10000 == 0 or (idx + 1) == total:
            print(f"Processadas {idx + 1}/{total} imagens...")


# --- Executando o export ---
base_output = "./MNIST/images"

export_mnist_to_folders(x_train, y_train, os.path.join(base_output, "train"))
export_mnist_to_folders(x_test, y_test, os.path.join(base_output, "test"))

print("Exportação concluída com sucesso!")

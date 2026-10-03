import pandas as pd
from torchgen import model
from src.dataset import MNISTDataset
from torch.utils.data import DataLoader

from train import Modelo

if __name__ == "__main__":
    df = pd.read_csv("./data/metadados_treino.csv")
    dataset = MNISTDataset(df)
    print(len(dataset))
    image, label = dataset[0]
    print(image)
    print(image.shape, label)

    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True
    )
    model = Modelo()
    for batch_images, batch_labels in loader:
        print(batch_images.shape, batch_labels.shape)
        logits = model(batch_images)
        print(logits.shape)
        print(logits[0])
        print(logits[0].argmax(dim=0))
        print(batch_labels[0])
        break

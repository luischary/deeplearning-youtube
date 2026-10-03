import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


def default_transform(image):
    return image / 255.0


class MNISTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=default_transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "path"]
        label = self.dataframe.loc[idx, "label"]
        # transforma em tensor
        label_t = torch.tensor(label, dtype=torch.long)
        image = read_image(img_path).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label_t

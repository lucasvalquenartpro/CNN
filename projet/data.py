import os

import pandas as pd
import torch
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torchvision.io import read_image


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, genre, label, transform=None):
        self.image_paths = image_paths
        self.genre = genre
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join('../train', self.image_paths[idx])
        img = read_image(path)
        genre = float(self.genre[idx])
        label = float(self.label[idx])
        if self.transform:
            img = self.transform(img)

        genre_tensor = torch.tensor(genre, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return (img, genre_tensor), label_tensor

def transform(istrain = True):
    if istrain:
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def split_train_val_data(df, test_size=0.2, random_state=42):
    df['stratify_col'] = df['label'].astype(str) + '_' + df['genre'].astype(str)

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df['stratify_col']
    )

    return train_idx, val_idx

def dataframe(filename):
    return pd.read_csv(filename, sep='\t', header=None, names=['img_paths', 'label', 'genre'])

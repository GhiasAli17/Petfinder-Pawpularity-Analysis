# src/data.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def build_transforms(img_size, aug_type, train=True):
    if aug_type == "lite":
        if train:
            t = [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            t = [transforms.Resize((img_size, img_size))]
    elif aug_type == "strong":
        if train:
            t = [
                 
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
                transforms.RandomAffine(20, shear=10, scale=(0.8, 1.2)),
                transforms.RandomPerspective(0.2, p=0.5),
               
            ]
        else:
            t = [transforms.Resize((img_size, img_size))]
    else:
        raise ValueError("Unknown aug_type: %s" % aug_type)
    # this is added for both train and valid and have default values as per ImageNet
    t += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(t)


class ImageOnlyDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        path = os.path.join(self.img_folder, img_id + ".jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = row["Pawpularity"]
        return img, y


class ImageTabDataset(Dataset):
    def __init__(self, df, img_folder, tab_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.tab_cols = tab_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        path = os.path.join(self.img_folder, img_id + ".jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = row[self.tab_cols].values.astype(np.float32)
        y = row["Pawpularity"]
        return img, tab, y

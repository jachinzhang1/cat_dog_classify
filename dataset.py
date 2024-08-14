import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        with open(path, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx].split('\t')[0]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = int(self.data[idx].split('\t')[1])
        return img, label

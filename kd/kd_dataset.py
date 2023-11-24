import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import datasets

from PIL import Image


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class SonarDataset(data.Dataset):
    def __init__(self, input_dir, label_dir=None, image_size=(160, 160), transform=None, stage=1):
        self.label_dir = label_dir
        self.input_dir = input_dir
        self.image_size = image_size
        self.image_files = os.listdir(input_dir)
        self.transform = transform
        self.stage = stage
        if stage == 2:
            assert self.label_dir is not None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.input_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')  # convert into gray scale
        image = image.resize(self.image_size)
        if self.transform is not None:
            image = self.transform(image)

        if self.stage == 2:
            filename = self.image_files[idx]
            suffix = "_DN"
            label_path = os.path.join(self.label_dir, filename[:-4] + suffix + filename[-4:])
            label = Image.open(label_path).convert('L')
            label = label.resize(self.image_size)
            if self.transform is not None:
                label = self.transform(label)
        else:
            label = image

        return image, label

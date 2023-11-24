import os
import cv2
import h5py
import numpy as np
import math
import torch.nn.functional as F

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset
import xml.etree.ElementTree as ET


@regist_dataset
class DEBRIS_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'train_dataset')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


@regist_dataset
class DEBRIS_Train_plus(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'test_dataset')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}



@regist_dataset
class DEBRIS_Test(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        self.dataset_path = os.path.join(self.dataset_dir, 'DEBRIS', 'test_dataset')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path

        for root, dirs, files in os.walk(self.dataset_path):
            self.img_paths = files
            self.img_paths.sort()
            break

        self.xml_folder = os.path.join(self.dataset_dir, 'DEBRIS', 'test_annotation')

        print('fetch {} samples for test'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, file_name), as_gray=True)
        noisy_img = noisy_img.unsqueeze(0)
        _, _, h, w = noisy_img.shape
        if h < 196:
            noisy_img = F.pad(noisy_img, (0, 0, 0, 196 - h), mode='reflect')
        if w < 196:
            noisy_img = F.pad(noisy_img, (0, 196 - w, 0, 0), mode='reflect')
        noisy_img = noisy_img.squeeze(0)

        return {'real_noisy': noisy_img, 'file_name': file_name[:-4]}


@regist_dataset
class prep_DEBRIS_Train(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):

        if self.crop_size[0] == 160:
            self.dataset_path = os.path.join(self.dataset_dir, 'prep/DEBRIS_Train')

        self.sub_folder = 'RN'
        assert os.path.exists(self.dataset_path), 'There is no dataset %s' % self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, self.sub_folder)):
            self.img_paths = files
            self.img_paths.sort()
            break

        print('fetch {} samples for training'.format(len(self.img_paths)))

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        noisy_img = self._load_img(os.path.join(self.dataset_path, self.sub_folder, file_name), as_gray=True)
        return {'real_noisy': noisy_img}


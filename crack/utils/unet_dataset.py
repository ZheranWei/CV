import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2 as cv

from utils.utils import cvtColor


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        file = open(images, 'r')
        image_ids = file.read().splitlines()
        file.close()
        #sfiles = os.listdir(mask_dir)
        for i in image_ids:
            img_file = os.path.join(image_dir, i+".jpg")
            mask_file = os.path.join(mask_dir, i+".png")
            self.images.append(img_file)
            self.masks.append(mask_file)



    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]

        img = cvtColor(cv.imread(image_path))
        img = np.float32(img) / 255.0
        img = np.transpose(img, [2, 0, 1])

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)

        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

        return sample
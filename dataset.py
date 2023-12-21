import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class PatchedImageDataset(Dataset):
    def __init__(self, image, patch_size):
        super().__init__()
        self.image = image
        self.patch_size = patch_size
        self.num_patches_h = (image.shape[1] - patch_size) // patch_size + 1
        self.num_patches_w = (image.shape[2] - patch_size) // patch_size + 1

    def __len__(self):
        return self.num_patches_h * self.num_patches_w

    def __getitem__(self, idx):
        i = idx // self.num_patches_w
        j = idx % self.num_patches_w
        patch_idx = [i, j]
        patch = self.image[
            :,
            i * self.patch_size : (i + 1) * self.patch_size,
            j * self.patch_size : (j + 1) * self.patch_size,
        ]
        return patch_idx, patch
    
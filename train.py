from torchvision.transforms import ToTensor, Normalize
from PIL import Image
from dataset import PatchedImageDataset
from torch.utils.data import DataLoader
from model import PatchModel
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from generate import generate_image

import math


def adjust_learning_rate(optimizer, epoch, lr, min_lr, num_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



to_tensor = ToTensor()
normalize = Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
image = Image.open("./images/4.png").convert("RGB").resize((28,28))
tensor_image = to_tensor(image)
tensor_image = normalize(tensor_image)
train_ds = PatchedImageDataset(tensor_image,7)

model = PatchModel(num_layers=2,hidden_dim=8,patch_dim=49*3)
total_params = 0
for param in model.parameters():
    total_params += param.numel()
print(f"Total Params: {total_params}")
print(f"Compression Rate: {total_params/2352}")
train_dl = DataLoader(train_ds,10,shuffle=True)
start_lr = 1e-3
min_lr = 1e-5
optimizer = torch.optim.AdamW(model.parameters(),start_lr)
epochs = 3000
loss_fn = nn.SmoothL1Loss()
epoch_losses = []
for epoch in range(epochs):
    losses = []
    for sample in train_dl:
        patch_idx,patches = sample
        flattened_patches = patches.view(patches.shape[0], -1)
        out = model(*patch_idx)
        loss = loss_fn(out,flattened_patches)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    lr = adjust_learning_rate(optimizer,epoch,start_lr,min_lr,epochs,epochs//10)
    epoch_losses.append(np.mean(losses))
plt.plot(epoch_losses)
plt.show()
out_img = generate_image(model,7,tensor_image.shape[1],tensor_image.shape[2])
out_img = out_img*torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)  + torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
plt.imshow(out_img.permute(1,2,0))
plt.show()


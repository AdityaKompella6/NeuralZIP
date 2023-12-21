import torch
import matplotlib.pyplot as plt
@torch.no_grad()
def generate_image(model,patch_size,height,width,num_patches_h,num_patches_w):
    out_image = torch.empty(3,height,width)
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            out_image[
            :,
            i * patch_size : (i + 1) * patch_size,
            j * patch_size : (j + 1) * patch_size,
        ] = model(torch.tensor([i]),torch.tensor([j])).view(3,patch_size,patch_size)
    return out_image


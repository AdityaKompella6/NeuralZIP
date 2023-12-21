import torch

@torch.no_grad()
def generate_image(model,patch_size,width,height,num_patches_w,num_patches_h):
    out_image = torch.empty(3,width,height)
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            out_image[
            :,
            i * patch_size : (i + 1) * patch_size,
            j * patch_size : (j + 1) * patch_size,
        ] = model(torch.tensor([i]),torch.tensor([j])).view(3,patch_size,patch_size)
    return out_image


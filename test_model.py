from torchvision.transforms import ToTensor
from PIL import Image
from dataset import PatchedImageDataset
from torch.utils.data import DataLoader
from model import PatchModel

to_tensor = ToTensor()
image = Image.open("./images/4.png").convert("RGB").resize((28,28))
tensor_image = to_tensor(image)
test_ds = PatchedImageDataset(tensor_image,7)

model = PatchModel()
test_dl = DataLoader(test_ds,10,shuffle=True)
for sample in test_dl:
    patch_idx,patch = sample
    print(patch_idx)
    out = model(*patch_idx)
    print(out.shape)

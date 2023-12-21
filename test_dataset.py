import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from dataset import PatchedImageDataset


to_tensor = ToTensor()
image = Image.open("./images/4.png").convert("RGB").resize((28,28))
plt.title("Original Image")
plt.imshow(image)
plt.show()
tensor_image = to_tensor(image)
test_ds = PatchedImageDataset(tensor_image,7)
print(len(test_ds))
for i in range(len(test_ds)):
    idx,patch = test_ds[i]
    plt.title(f"{idx}")
    plt.imshow(patch.permute(1,2,0))
    plt.show()



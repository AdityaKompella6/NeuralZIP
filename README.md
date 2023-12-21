# NeuralZIP
Recreating linux zip for directories of images using neural networks

The images we will be testing our method on is MNIST.

Each image is represented by 3*28*28 numbers so our technique must 
have storage that is more efficient than 2352 floats.

We will be tackling this problem by patching up the images and learning a representation 
from patch indeces to the pixels of the patch.

Usage:

python zip.py --input_path (Path to Image File) --output_path (Where to save zipped file) --patch_size (what size patches you want to use to make the model generate, Default: 10) --num_epochs (How many epochs to train)

python unzip.py --input_path (Path to zip file) --output_path (Where to save uncompressed Image File) 

Run zip.py and unzip.py on the car image with patch_size = 20 to see a 10x compression!
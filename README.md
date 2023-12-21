# NeuralZIP
Recreating linux zip for directories of images using neural networks

The images we will be testing our method on is MNIST.

Each image is represented by 3*28*28 numbers so our technique must 
have storage that is more efficient than 2352 floats.

We will be tackling this problem by patching up the images and learning a representation 
from patch indeces to the pixels of the patch.

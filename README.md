# Simple-Image-Classifier


Classifies an image based on a data set provided by the user.

## How to use:
Create a central folder to hold all of your images that will be used for classification. Inside, organize the images so that each image of a certain class is in its own folder with the classification name. Example: Create a central folder called "dataset" in the same folder as the python files. You plan to classify pokemon. Inside the "dataset" folder, create a folder called "Charizard" containing all of your Charizard images.

## In the command line:
Call the code as follows if you want to compute new bases for classification and classify an image.

The code below will create new bases for classification using the images in dataset/ and then try to classify jigglypuff.jpg...
python.exe identify.py dataset/ jigglypuff.jpg

If you already computed the bases and do not wish to re-compute, change the directory to "load"...
python.exe identify.py load jigglypuff.jpg

Inside the identify.py folder, there are a few global variables to tweak.
NUM_COLUMNS changes the number of vectors in each basis.
SIZE changes the scaled down size of the pictures. 28 means 28x28px images. This also makes computing much easier. The tested image will also be scaled down.

That's it.
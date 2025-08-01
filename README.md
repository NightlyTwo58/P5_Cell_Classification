# Natural Resource Exports & HDI Analysis

## Project Overview
![Dog from testing set](data/test/dogs_set/dog.4001.jpg)
This project is a Convolutional Neural Network (CNN) designed to classify images of cats and dogs. The model's architecture is a custom-built deep network with 4 convolutional blocks, a BatchNormalization layer after each convolution, and a ReLU activation function.
The network uses data augmentation with a 10% rotation and 10% zoom to increase the dataset's size and prevent overfitting. The convolutional blocks reduce the image size from 250x250 pixels to 13x13 pixels before feeding them into two large dense layers, each with 2048 neurons and a high 50% dropout rate for regularization.
The model is trained using the Adam optimizer with a low learning rate of 1e<sup>−4</sup>. A ReduceLROnPlateau callback monitors the validation loss and reduces the learning rate by a factor of 0.2 if no improvement is seen for 4 epochs, with a minimum learning rate of 1e<sup>−6</sup>. The model is trained for 9 (arbitrary) epochs, with a ModelCheckpoint saving the best weights.
![Cat from testing set](data/test/cats_set/cat.4001.jpg)

### Installation
Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Authors
* **Benjamin Luo**
* **Richard Cai**
# Handwritten Digits Recognition
This project is a simple implementation of a neural network to recognize handwritten digits. The neural network is trained on the MNIST dataset, which is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. The neural network is implemented in Python using the PyTorch library.
Thanks to [linuslau](https://github.com/linuslau/NumberRecognition) for the idea and the code.

## Requirements
```shell
conda create -n pytorch python=3.9.18
conda activate pytorch

# install pytorch, the version is for CUDA 12.1
# from https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
pip3 install -r requirements.txt
```

## Usage
```shell
python3 main.py
```

The `config.json` file contains the configuration for the neural network. 
- `mode`: `default`, `train`, `evaluate` or `inference`. If `default`, the neural network will be trained, evaluated and used for inference. If `train`, the neural network will be trained. If `evaluate`, the neural network will be evaluated. If `inference`, the neural network will be used for inference. If `inference`, the neural network will be used to make predictions, and the images will be saved to the `images.png` file.
- `model`: `dense` or `conv`. If `dense`, a dense neural network will be used. If `conv`, a convolutional neural network will be used.
- `data`: the location of the MNIST dataset. **If the dataset is not found, it will be downloaded to this location**.
- `epochs`: the number of epochs to train the neural network for.
- `batch_size`: the batch size to use for training.
- `lr`: the learning rate to use for training.

## Results
The neural network is trained on the MNIST dataset for 10 epochs with a batch size of 64 and a learning rate of 0.001. The dense neural network achieves an accuracy of **97.85%** on the test set. The convolutional neural network achieves an accuracy of **99.13%** on the test set.
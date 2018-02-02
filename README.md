# Simple CNN for Handwritten numbers

## Getting started

Install Keras with TensorFlow or Theano backend. (Tested with Theano only)

## Data source

MNIST datasource for handwritten numbers (60,000 images, 28x28 pixels) 

```python
from keras.datasets import mnist
data = mnist.load_data()
```

## Accuracy

Over 99% accuracy on the famous MNIST dataset


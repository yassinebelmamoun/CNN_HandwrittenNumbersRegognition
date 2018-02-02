""" CNN learn Handwritten number """
import numpy as np
# Linear stack of neural network layers (for feed-forward CNN)
from keras.models import Sequential
# "core" layers of Neural Networks 
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers to train on image data
from keras.layers import Convolution2D, MaxPooling2D
# Transform our data
from keras.utils import np_utils
# Load our data
from keras.datasets import mnist
# Plot the image
from matplotlib import pyplot as plt
# Reproductibility
np.random.seed(123)
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Pre-processing our data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Let's start the model architecture
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# the weights from the Convolution layers must be flattened (made 1-dimensional)
# before passing them to the fully connected Dense layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)


# Build a shallow neural network to classify MNIST digits

# set seed for reproducibility
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#print(X_train.shape)
#print(X_test.shape)

# preprocess data
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train /= 255
X_test /= 255
#print(X_train[0])
n_classes = 10
Y_train = keras.utils.to_categorical(Y_train, n_classes)
Y_test = keras.utils.to_categorical(Y_test, n_classes)
#print(Y_train[0])

# design neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# configure model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

# train
model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, Y_test))
weights = model.get_weights()
for i in range(np.shape(weights)[0]):
    print(np.shape(weights[i]))


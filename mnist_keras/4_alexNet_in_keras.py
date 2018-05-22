# alexnet
# we're going to classify Oxford Flowers

# set seed
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

# load and preprocess data
import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True)

# design neural network architecture
model = Sequential()
model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(17, activation='softmax'))
model.summary()

# configure model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# tensorboard = TensorBoard(log_dir='./logs/alexnet', write_graph=True, write_images=True, histogram_freq=0)
# tensorboard --logdir=/Users/sina/logs/ --port 6006

# train
# model.fit(X, Y, batch_size=64, epochs=1, verbose=1, validation_split=0.1, shuffle=True, callbacks=[tensorboard])
model.fit(X, Y, batch_size=64, epochs=5, verbose=1, validation_split=0.1, shuffle=True)

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Dropout, Flatten
from keras.utils import to_categorical

# -- k49 data -- 49 classifications
x_train = np.load('data/k49-train-imgs.npz')['arr_0']
y_train = np.load('data/k49-train-labels.npz')['arr_0']

x_test = np.load('data/k49-test-imgs.npz')['arr_0']
y_test = np.load('data/k49-test-labels.npz')['arr_0']
classifications = 49
# -----

# # -- basic kmnist dataset -- 10 classifications
# x_train = np.load('data/kmnist-train-imgs.npz')['arr_0']
# y_train = np.load('data/kmnist-train-labels.npz')['arr_0']

# x_test = np.load('data/kmnist-test-imgs.npz')['arr_0']
# y_test = np.load('data/kmnist-test-labels.npz')['arr_0']
# classifications = 10
# # -----

# -- train data format
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

x_train = x_train.reshape(x_train.shape[0], *input_shape)

y_train = to_categorical(y_train, classifications)

# -- test data format

x_test = x_test.astype('float32')
x_test /= 255

x_test = x_test.reshape(x_test.shape[0], *input_shape)

y_test = to_categorical(y_test, classifications)
# ---

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classifications, activation='softmax'))

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, verbose=1,
          batch_size=128, epochs=12,
          validation_data=(x_test, y_test))

model.save('model.h5')


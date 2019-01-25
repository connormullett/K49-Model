
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Dropout, Flatten
from keras.utils import to_categorical

x_train = np.load('data/k49-train-imgs.npz')['arr_0']
y_train = np.load('data/k49-train-labels.npz')['arr_0']

# -- data format
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

x_train = x_train.reshape(x_train.shape[0], *input_shape)

y_train = to_categorical(y_train, 49)
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
model.add(Dense(49, activation='softmax'))

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=1)

model.save('model.h5')


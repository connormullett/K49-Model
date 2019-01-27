
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from tensorflow import keras

import pandas as pd

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
class_map = pd.read_csv('data/k49_classmap.csv')

# -- Data format
x_test = np.load('data/k49-test-imgs.npz')['arr_0']
y_test = np.load('data/k49-test-labels.npz')['arr_0']

classifications = 49

x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0], *input_shape)
y_test = to_categorical(y_test, classifications)

model = keras.models.load_model('model.h5')

loss, acc = model.evaluate(x_test, y_test)
print('acc: {}\nloss: {}'.format(100 * acc, loss))

predictions = model.predict(x_test[:1000], verbose=1)

for i in range(10):
    prediction_test = np.argmax(predictions[i])
    print(class_map.loc[prediction_test, 'char'])
    actual = x_test[i]
    pixels = actual.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


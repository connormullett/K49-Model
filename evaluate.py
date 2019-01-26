
import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from tensorflow import keras

import pandas as pd

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
class_map = pd.read_csv('data/kmnist_classmap.csv')

x_test = np.load('data/kmnist-test-imgs.npz')['arr_0']
y_test = np.load('data/kmnist-test-labels.npz')['arr_0']

classifications = 10

x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0], *input_shape)
y_test = to_categorical(y_test, classifications)

model = keras.models.load_model('kmnist_model.h5')

predictions = model.predict(x_test[:1000], verbose=1)
prediction_test = predictions[0]

loss, acc = model.evaluate(x_test, y_test)
print('acc: {}'.format(100 * acc))


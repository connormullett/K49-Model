
import tensorflow as tf
import numpy as np

from tensorflow import keras


x_test = np.load('data/k49-test-imgs.npz')['arr_0']
y_test = np.load('data/k49-test-labels.npz')['arr_0']

model = keras.models.load_model('model.h5')

loss, acc = model.evaluate(x_test, y_test)
print('acc: {}'.format(100 * acc))


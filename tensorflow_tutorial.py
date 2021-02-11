# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:27:11 2020

@author: Alok Garg
"""
import tensorflow as tf


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(11,input_shape= (14,1),activation='softmax')])


if __name__ == '__main__':
    print('learning tensorflow')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
   # x_train, x_test = x_train / 255.0, x_test / 255.0
   # check_two = x_train[:2]
   # check_two_flatten = np.flatten(check_two)
   x_train, x_test = x_train / 255.0, x_test / 255.0

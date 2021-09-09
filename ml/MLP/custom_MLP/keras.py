import numpy as np
import tensorflow as tf


# input - hidden layer
w1 = np.random.randn(6, 15)
b1 = np.random.randn(6)
f1 = "sgimoid"
# hidden - output layer
w2 = np.random.randn(3, 6)
b2 = np.random.randn(3)
f2 = "softmax"

#------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Dense(6, input_shape=(15,), activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='softmax')
])
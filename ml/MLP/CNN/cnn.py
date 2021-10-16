import tensorflow as tf

from data_mnist import mnist2Data

x_train, y_train, x_test, y_test = mnist2Data()

x_train, x_test = x_train/255.0, x_test/255.0
#CNN의 경우 input shape에 채널까지 들어가야함(흑백 1채널, rgb 3채널)
x_train = x_train.reshape(len(x_train), x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(len(x_test), x_test.shape[1], x_test.shape[2], 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
print(test_acc)
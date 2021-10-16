import tensorflow as tf
import numpy as np


#실제 결과로 매핑할 데이터
x_train = np.array([ [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]# 0
                    ,[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]# 1
                    ,[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]# 2
                    ,[1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]# 2
                    ,[1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]# 2
                    ,[1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]# 2
                    ], dtype="uint8")#* 255
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype="uint8")


zero = np.array([    1, 1, 1
                    ,1, 0, 1
                    ,1, 0, 1
                    ,1, 0, 1
                    ,1, 1, 1], dtype="uint8")# * 255
one = np.array([    0, 1, 0
                   ,1, 1, 0
                   ,0, 1, 0
                   ,0, 1, 0
                   ,1, 1, 1], dtype="uint8")# * 255
two = np.array([    1, 1, 1
                   ,0, 0, 1
                   ,1, 1, 1
                   ,1, 0, 0
                   ,1, 1, 1], dtype="uint8")# * 255
x_test = np.array([zero, one, two])


#보통 이미지의 경우 (데이터 갯수, height, width)3차원
#CNN의 경우 input shape에 채널까지 들어가야함(흑백 1채널, rgb 3채널)

#해당 케이스는 (데이터 갯수, 15)2차원 이어서 차원변환 해야함
#(갯수, 5, 3)이 아닌 (갯수, 5, 3, 1)로 채널까지 만들어줘야함
x_train = x_train.reshape(len(x_train), 5, 3, 1)
x_test = x_test.reshape(len(x_test), 5, 3, 1)

#create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(x_train, y_train, epochs=500)

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
print(model.predict(x_test))
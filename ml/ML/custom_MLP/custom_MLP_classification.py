import numpy as np

def weighted_sum(x, w, b):
    return np.sum(x * w, axis = 1) + b

#sigmoid activate function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_dif(x):
    return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

#실제 결과로 매핑할 데이터
train_x = np.array([ [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
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
train_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype="uint8")

# print(train_x)
# print(train_y)


# input - hidden layer
w1 = np.random.randn(6, 15)
# print("weight1\n", w1)
b1 = np.random.randn(6)
# print("bias1\n", b1)


# hidden - output layer
w2 = np.random.randn(3, 6)
# print("weight2\n", w2)
b2 = np.random.randn(3)
# print("bias2\n", b2)


epoch = 1000
learnRate = 0.7
mse = []
Error = None
for index_epoch in range(epoch):
    print(index_epoch+1, "번째 학습입니다.", index_epoch+1 , "/", epoch)
    err = []
    for index_train in range(len(train_x)):
        # hidden to output Layer
        # FeedForword(순전파)
        # input to hidden Layer
        i2hLayerNet = weighted_sum(train_x[index_train], w1, b1)
        i2hLayer = sigmoid(i2hLayerNet)
        # hidden to output Layer
        h2oLayerNet = weighted_sum(i2hLayer, w2, b2)
        h2oLayer = softmax(h2oLayerNet)

        # Error
        Error = np.sum(h2oLayer[train_y[index_train]])-1
        err.append(Error)

        # Back-Propagation(역전파)
        # Cost = (loss function + softmax)differential
        # Softmax with Loss
        Cost = np.copy(h2oLayer)
        Cost[train_y[index_train]] -= 1

        # hidden to output Layer
        a2 = Cost
        b2 = b2 - learnRate * a2
        w2 = w2 - (learnRate * a2.reshape(a2.shape[0], 1) * i2hLayer)

        # input to hidden Layer
        a1 = np.sum(a2.reshape(a2.shape[0], 1) * sigmoid_dif(i2hLayerNet), axis=0)
        b1 = b1 - learnRate * a1
        w1 = w1 - (learnRate * a1.reshape(a1.shape[0], 1) * train_x[index_train])

    #학습완료
    print("mse", sum(err))


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
test = [zero, one, two]
for index in range(len(test)):
    # input to hidden Layer
    i2hLayer = sigmoid(weighted_sum(test[index], w1, b1))

    # hidden to output Layer
    h2oLayer = softmax(weighted_sum(i2hLayer, w2, b2))

    #소숫접3이하 버리기
    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
    print(h2oLayer)

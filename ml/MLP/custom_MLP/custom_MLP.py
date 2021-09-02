import numpy as np
from matplotlib import pyplot as plt



def weighted_sum(x, w, b):
    return np.sum(x * w) + b

#sigmoid activate function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    m = np.max(x)
    exp_x = np.exp(x-m)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

#실제 결과로 매핑할 데이터
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
# print("zero\n", zero)
# print("one\n", one)

train_x = np.array([ [1, 1, 0,1, 0, 1,1, 0, 1,1, 0, 1,1, 1, 1]# 0
                    ,[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]# 0
                    ,[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]# 0
                    ,[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]# 1
                    ,[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]# 1
                    ], dtype="uint8")# * 255
train_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="uint8")# * 255

# print(train_x)
# print(train_y)


# input - hidden layer
w1 = np.random.randn(15)
# print("weight1\n", w1)
b1 = np.random.randn(6)
# print("bias1\n", b1)


# hidden - output layer
w2 = np.random.randn(6)
# print("weight2\n", w2)
b2 = np.random.randn(1)
# print("bias2\n", b2)

epoch = 1000
learnRate = 0.7
mse = []
Error = None
for index_epoch in range(epoch):
    print(index_epoch+1, "번째 학습입니다.", index_epoch+1 , "/", epoch)
    err = []
    for index_train in range(len(train_x)):
        # FeedForword(순전파)

        # input to hidden Layer
        i2hLayer = sigmoid(weighted_sum(train_x[index_train], w1, b1))
        # hidden to output Layer
        h2oLayer = sigmoid(weighted_sum(i2hLayer, w2, b2))

        #error
        Error = ((train_y[index_train] - h2oLayer)**2)/2
        # result = np.append(result, h2oLayer)

        #Back-Propagation(역전파)
        # hidden to output Layer
        a2 = -(train_y[index_train] - h2oLayer) * h2oLayer * (1 - h2oLayer)
        b2 = b2 - learnRate * a2
        w2 = w2 - (learnRate * a2 * i2hLayer)

        # input to hidden Layer
        a1 = np.sum(a2 * i2hLayer * (1 - i2hLayer))
        b1 = b1 - learnRate * a1
        w1 = w1 - (learnRate * a1 * train_x[index_train])
        err.append(Error)
        print("mse", Error)
    #학습완료
    print("mse", sum(err))



# plt.xlabel('EPOCH')
# plt.ylabel('MSE')
# plt.title('MLP TEST')
# plt.plot(mse)
# plt.show()



test = [zero, one]
for index in range(len(test)):
    # input to hidden Layer
    i2hLayer = sigmoid(weighted_sum(test[index], w1, b1))

    # hidden to output Layer
    h2oLayer = sigmoid(weighted_sum(i2hLayer, w2, b2))

    print(h2oLayer)

import pandas as pd
import numpy as np


# iris.csv(150건)
# Sepal Length(꽃받침길이)
# Sepal Width(꽃받침 너비)
# Petal Length(꽃잎 길이)
# Petal Width(꽃잎 너비)
# Speicies(품종)(0~3으로 카테고리화 함)

# train(150건) == test(150건)
def irisData(col=-1):
    data = pd.read_csv("iris.csv").values
    a, b = np.unique(data[:, -1], return_inverse=True)
    data[:, -1] = b
    data = data[:, 1:]
    data = data.astype(np.float32)
    y = data[:, col]
    x = np.delete(data, col, axis=1)
    return x, y, np.copy(x), np.copy(y)


# 사용법
# default(4)species
x_train, y_train, x_test, y_test = irisData()

# (3)Petal Width
x_train, y_train, x_test, y_test = irisData(3)

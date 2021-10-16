import numpy as np
import matplotlib.pyplot as plt


def LeakyReLU(x, a):
    return np.where(x > 0, x, x * a)

x = np.arange(-5.0, 5.0, 0.1)
y = LeakyReLU(x, 0.1)
plt.plot(x, y)
plt.ylim(-1, 5.0)
plt.show()
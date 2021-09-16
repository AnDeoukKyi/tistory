import cv2
import numpy as np


def te(img):
    im = np.copy(img)
    arr = []
    for i in range(img.shape[0]):
        ls = []
        for j in range(img.shape[1] - 1):
            ls.append(int(np.average(img[i, j:j+2])))
        arr.append(np.array(ls))
    return np.array(arr, dtype="uint8")

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("origin", img)
for i in range(30):
    img = te(img)

cv2.imshow(str(i), img)
cv2.waitKey()






import cv2
import numpy as np

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#width 2배
M = np.float32([[2, 0, 0], [0, 1, 0]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("widthx2", shifted)


#height 2배
M = np.float32([[1, 0, 0], [0, 2, 0]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("heightx2", shifted)


#right +50
M = np.float32([[1, 0, 50], [0, 1, 0]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("right+50", shifted)


#bottom +50
M = np.float32([[1, 0, 0], [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("bottom+50", shifted)

cv2.waitKey(0)
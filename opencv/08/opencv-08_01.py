import cv2
import numpy as np

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#BGR
(B, G, R) = cv2.split(img)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
# 흰색 배경도 R값은 255임
# White(255, 255, 255)

zeros = np.zeros(img.shape[:2], dtype="uint8")
merged = cv2.merge([B, zeros, R])
cv2.imshow("Merged", merged)

ones = np.ones(img.shape[:2], dtype="uint8") * 255
merged1 = cv2.merge([B, ones, R])
cv2.imshow("Merged1", merged1)

cv2.waitKey(0)
import cv2
import numpy as np

add = cv2.add(np.uint8([200]), np.uint8([100]))
print("cv2", add)#cv2 [[255]]
add = np.uint8([200]) + np.uint8([100])
print("np", add)#np [44]


img = cv2.imread("image.png")
cv2.imshow("origin", img)

M = np.ones(img.shape, dtype="uint8") * 100
# M = np.zeros(img.shape, dtype="uint8")
# M.fill(100)


#Light
added = cv2.add(img, M)
cv2.imshow("Light", added)


#Dark
sub = cv2.subtract(img, M)
cv2.imshow("Dark", sub)

cv2.waitKey(0)


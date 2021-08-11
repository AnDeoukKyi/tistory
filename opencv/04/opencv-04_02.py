import cv2
import imutils

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#rotate 90도
rotated = imutils.rotate(img, 90)
cv2.imshow("rotate 90", rotated)

cv2.waitKey(0)

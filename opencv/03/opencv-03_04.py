import cv2
import imutils

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#width 150
shifted = imutils.resize(img, width=150)
cv2.imshow("width 150", shifted)

#width 200 hegith 100
shifted = imutils.resize(img, width=200, height=100)
cv2.imshow("w200h100", shifted)

cv2.waitKey(0)


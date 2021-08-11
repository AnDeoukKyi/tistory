import cv2
import imutils

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#right +50
shifted = imutils.translate(img, 50, 0)
cv2.imshow("right+50", shifted)

#bottom +50
shifted = imutils.translate(img, 0, 50)
cv2.imshow("bottom+50", shifted)


cv2.waitKey(0)


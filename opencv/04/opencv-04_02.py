import cv2
import imutils

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#rotate 45도
rotated = imutils.rotate(img, 45)
cv2.imshow("rotate 45", rotated)


#rotate bound 45도
rotated = imutils.rotate_bound(img, 45)
cv2.imshow("rotate bound 45", rotated)


cv2.waitKey(0)

import cv2

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#horizontal
flipped = cv2.flip(img, 1)
cv2.imshow("horizontal", flipped)

#vertical
flipped = cv2.flip(img, -1)
cv2.imshow("vertical", flipped)

cv2.waitKey(0)


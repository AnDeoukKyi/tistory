import cv2

img = cv2.imread("image1.png")

#원본으로 확대 225x225
shifted = cv2.resize(img, (15, 15), interpolation=cv2.INTER_AREA)
cv2.imwrite("AREA.png", shifted)

#원본으로 확대 225x225
shifted = cv2.resize(img, (15, 15), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("CUBIC.png", shifted)

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#50x50축소
img = cv2.resize(img, (50, 50))
cv2.imshow("reduce 50x50", img)

#원본으로 확대 225x225
shifted = cv2.resize(img, (225, 225))
cv2.imshow("255", shifted)

cv2.waitKey(0)


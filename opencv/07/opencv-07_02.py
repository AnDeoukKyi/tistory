import cv2
import numpy as np

img = cv2.imread("image.png")
cv2.imshow("origin", img)


#마스킹 사각형
mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.rectangle(mask, (40, 100), (150, 200), 255, -1)
cv2.imshow("Rectangle", mask)

#AND연산
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Mask", masked)

cv2.waitKey(0)
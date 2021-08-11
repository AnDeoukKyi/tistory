import cv2

img = cv2.imread("image.png")
cv2.imshow("origin", img)

(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)

#중심rotate 45도
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("C_rotate 45", rotated)
print(M)

#원점rotate 45도
M = cv2.getRotationMatrix2D((0, 0), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("0_rotate 45", rotated)
print(M)

#원점rotate 45도
M = cv2.getRotationMatrix2D((0, 0), 45, 0.5)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("0S_rotate 45", rotated)

cv2.waitKey(0)
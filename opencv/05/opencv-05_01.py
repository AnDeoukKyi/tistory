import cv2

img = cv2.imread("image.png")
cv2.imshow("origin", img)

#ROI1
ROI = img[100:150,50:150]
cv2.imshow("ROI1", ROI)

#ROI2
ROI = img[:150,50:]
cv2.imshow("ROI2", ROI)

cv2.waitKey(0)
import cv2

img = cv2.imread("car_number.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("origin", img)

# 노이즈 제거
ret, img = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", img)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
cv2.imshow("DILATE", img)
img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
cv2.imshow("ERODE", img)




cv2.waitKey(0)
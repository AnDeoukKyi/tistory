import cv2

img = cv2.imread("image.png")

#직선 그리기
cv2.line(img, (0, 0), (100, 100), (0, 255, 0))

#원 그리기
cv2.circle(img, (100, 100), 15, (0, 0, 255), 2)
cv2.circle(img, (50, 50), 10, (0, 0, 255), -1)

#사각형 그리기
cv2.rectangle(img, (170, 150), (200, 200), (255, 0, 0), 2)
cv2.rectangle(img, (0, 150), (100, 200), (255, 0, 0), -1)

cv2.imshow("02_01 draw", img)
cv2.waitKey(0)#키 입력
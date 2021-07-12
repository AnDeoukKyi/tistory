import cv2

# 원본 이미지 읽어서 출력
img = cv2.imread("image.png")

cv2.imshow("01_01 original", img)
cv2.waitKey(0)#키 입력

# 흑백 이미지로 읽어서 출력
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("01_01 grayscale", img)
cv2.waitKey(0)#키 입력
import cv2

# 원본 이미지 읽어서 출력
img = cv2.imread("image.png")

cv2.imshow("01_02 original", img)
cv2.waitKey(0)#키 입력

# 흑백 이미지로 읽어서 출력
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("01_02 grayscale", img)
cv2.waitKey(0)#키 입력

# newImage.png로 저장
cv2.imwrite("newimage.png", img)
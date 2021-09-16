import cv2
import numpy as np
import random

def create_captchaImg(filePath, size, num):
    src = np.zeros(size, dtype="uint8")
    inputImg = cv2.imread("resource/" + str(num) + ".png", cv2.IMREAD_GRAYSCALE)

    x, y = random.randrange(0, size[1] - inputImg.shape[1]), random.randrange(0, size[0] - inputImg.shape[0])
    roi = src[y:y + inputImg.shape[0], x:x + inputImg.shape[1]]
    roi = cv2.add(roi, inputImg)
    src[y:y + inputImg.shape[0], x:x + inputImg.shape[1]] = roi

    cv2.imshow("A", src)
    cv2.waitKey()
    #파일 갯수 체크해야됨
    cv2.imwrite(filePath + "/" + str(num) + ".png", src)


create_captchaImg("img", (100, 150), random.randrange(0, 10))


import cv2
import numpy as np

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("origin", img)







cv2.waitKey(0)
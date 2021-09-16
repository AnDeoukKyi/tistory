import cv2
import numpy as np

def create_Number(filePath, num):
    src = np.zeros((200 ,200), dtype="uint8")
    cv2.putText(src, str(num), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, 255, 1, cv2.LINE_AA)

    # cv2.imshow("A", src)
    # cv2.waitKey()
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    for x, y, w, h, cnt in stats:
        if(h, w) < src.shape:
            # cv2.rectangle(palette, (x, y, w, h), 255, 1)
            src = src[y:y+h, x:x+w]
            cv2.imwrite(filePath + "/" + str(num) + ".png", src)
            break

    # cv2.imshow("A", src)
    # cv2.waitKey()
for i in range(10):
    create_Number("resource", i)


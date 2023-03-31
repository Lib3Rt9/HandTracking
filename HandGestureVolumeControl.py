import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()                       # check success of video capture
    img = cv2.flip(img, 1)

    cv2.imshow("img", img)
    cv2.waitKey(1)
# test run the module Hand Tracking in another project

import cv2
import time

# import Hand Tracking Module
import HandTrackingModule as htm

pTime = 0  # current time
cTime = 0  # previous time

# create video object
cap = cv2.VideoCapture(0)  # use webcam 0
detector = htm.handDetector()

while True:
    success, img = cap.read()  # get the frames
    img = detector.findHands(cv2.flip(img, 1))              # also flip the image
    # img = cv2.flip(img, 1)
    lmksList = detector.findPosition(img,
                                     # draw=False
                                     )

    if len(lmksList) != 0:
        print(lmksList[4])

    # see fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,  # put it into img
                str(int(fps)),  # str return decimal value, convert to int
                (10, 80),  # set position
                cv2.FONT_HERSHEY_PLAIN,  # set Font
                3,  # font scale
                (0, 0, 255),  # color value (BGR) (let's take red)
                3)  # text thickness

    # let's see the webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # waitKey is used to display a frame for 1ms, after which display will be automatically closed

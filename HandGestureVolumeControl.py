import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
#______________
import pycaw                                        # import library
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#--------------

######### PARAMETERS ######################
W_CAM, H_CAM = 640, 480                             # width camera, height camera captured
###########################################


cap = cv2.VideoCapture(0)
cap.set(3, W_CAM)                                   # id no.3 = width
cap.set(4, H_CAM)                                   # id no.4 = height
pTime = 0                                           # previous time

detector = htm.handDetector()                       # create an object detector


# components from pycaw library
# init
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()

volRange = volume.GetVolumeRange()                  # volume range from -65.25 to 0.0 (-65.25 - 0.0)
# volume.SetMasterVolumeLevel(-20.0, None)

# get the min max range
minVolume = volRange[0]
maxVolume = volRange[1]












while True:
    success, img = cap.read()                       # check success of video capture
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmrkList = detector.findPosition(img, draw=True)
    if len(lmrkList) != 0:                                  # check if nothing is found
            # print(lmrkList[4], lmrkList[8])

            # create vars for center points of landmarks
            x1, y1 = lmrkList[4][1], lmrkList[4][2]         # landmark at point no.4
            x2, y2 = lmrkList[8][1], lmrkList[8][2]         # landmark at point no.8

            # get the center of the line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # make sure using correct landmarks, create a circle
            cv2.circle(img,                             # circle 1
                    (x1, y1),                                   # center point
                    10,                                         # radius
                    (255, 0, 255),                              # color
                    cv2.FILLED)                                 # fill
            cv2.circle(img,                             # circle 2
                    (x2, y2),                                   # center point
                    10,                                         # radius
                    (255, 0, 255),                              # color
                    cv2.FILLED)                                 # fill
            
            cv2.line(img,                               # create a line between circle 1 and 2
                    (x1, y1),                                   # point 1
                    (x2, y2),                                   # point 2
                    (0, 255, 0),                              # color
                    3)                                          # thickness
            
            cv2.circle(img,                             # circle middle
                    (cx, cy),                                   # center point
                    8,                                          # radius
                    (255, 0, 0),                                # color
                    cv2.FILLED)                                 # fill


            # find the length of the line between 2 points above -> change volume base on that length
            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)


            # hand range    from    50      to  250     (50      -  250)
            # volume range  from    -65.25  to  0.0     (-65.25  -  0.0)

            # convert hand range to volume range
            vol = np.interp(length,                             # value to convert
                            [50, 250],                          # range value to convert
                            [minVolume, maxVolume])             # convert to range value

            print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)


            if length < 50:
                cv2.circle(img,                             # circle 2
                    (cx, cy),                                   # center point
                    8,                                          # radius
                    (0, 0, 255),                                # color
                    cv2.FILLED)                                 # fill

        






    # see frame per second (fps)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,                                            # put it into img
                    f"FPS: {int(fps)}",                         # str return decimal value, convert to int
                    (10, 60),                                   # set position
                    cv2.FONT_HERSHEY_SIMPLEX,                   # set Font
                    1,                                          # font scale
                    (255, 0, 0),                                # color value (BGR) (let's take red)
                    3) 


    cv2.imshow("img", img)
    cv2.waitKey(1)
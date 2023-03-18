import cv2
import mediapipe as mp
import time                     # to check frame rate

# create video object
cap = cv2.VideoCapture(0)       # use webcam 0

# hand detection module
# 1st - create an object from class Hands
mpHands = mp.solutions.hands    # formality - have to do before using module "hands" of mediapipe
hands = mpHands.Hands()         # no need parameters - the "Hands" has them all - follow the "Hands" to see params
mpDraw = mp.solutions.drawing_utils

pTime = 0                       # current time
cTime = 0                       # previous time

while True:
    success, img = cap.read()   # get the frames
    img = cv2.flip(img, 1)      # flip the image from front camera

    # let's detect the hand

    # send in rgb image to "hands" object
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # convert to RGB since "hands" uses only BGR
    results = hands.process(imgRGB)                     # call "hands" and process the frame and get the results

    # extract information of object "results" - extract 'multiple hands'
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:                    # use "multi_hand_landmarks" to check if there is something detected
        for handLmks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLmks.landmark):
                # print(id, lm)

                h, w, c = img.shape                     # width and height of the img
                cx, cy = int(lm.x*w), int(lm.y*h)       # convert decimal values to pixel values

                # print(id, cx, cy)
                # if id == 4:                             # id = location of the landmark
                    # draw the circle
                cv2.circle(img,                     # draw circle for easier figuring the in-consider landmark
                           (cx, cy),                # coordinate of the landmark in pixels
                           6,                       # radius of circle
                           (0, 255, 0),             # color value (BGR) (let's take purple)
                           cv2.FILLED)              # fill the drawn circle

            mpDraw.draw_landmarks(img, handLmks, mpHands.HAND_CONNECTIONS)

    # see fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,                        # put it into img
                str(int(fps)),              # str return decimal value, convert to int
                (10, 80),                   # set position
                cv2.FONT_HERSHEY_PLAIN,     # set Font
                3,                          # font scale
                (0, 0, 255),                # color value (BGR) (let's take red)
                3)                          # text thickness

    # let's see the webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)              # waitKey is used to display a frame for 1ms, after which display will be automatically closed


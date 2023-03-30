import cv2
import mediapipe as mp
import time                                                     # to check frame rate


# hand detection module
class handDetector():
    def __init__(self,
                 mode=False,                                    # static_image_mode=False,
                 maxHands=2,                                    # max_num_hands=2,
                 modelComplex=1,                                # model_complexity=1,
                 detexCon=0.5,                                  # min_detection_confidence=0.5,
                 trackCon=0.5):                                 # min_tracking_confidence=0.5
                
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detexCon = detexCon
        self.trackCon = trackCon

        # 1st - create an object from class Hands
        self.mpHands = mp.solutions.hands                       # formality - have to do before using module "hands" of mediapipe
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.modelComplex,
                                        self.detexCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,
                  img,                                          # to find the hand
                  draw=True):                                   # draw or not the detections
        # let's detect the hand

        # send in rgb image to "hands" object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # convert to RGB since "hands" uses only BGR
        self.results = self.hands.process(imgRGB)               # call "hands" and process the frame and get the results

        # extract information of object "results" - extract 'multiple hands'
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:                   # use "multi_hand_landmarks" to check if there is something detected
            for handLmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLmks,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,                                      # to get position values of landmarks
                     img,                                       # need the width and height -> need shape
                     handNo=0,                                  # information for hand number #n
                     draw=True):                                # draw or not

        lmksList = []                                           # Landmarks list - contains all landmarks positions
        
        # check if the landmarks are detected
        if self.results.multi_hand_landmarks:
            considerHands = self.results.multi_hand_landmarks[handNo]       # point to the specific hand which in consideration

            for id, lm in enumerate(considerHands.landmark):
                # print(id, lm)

                h, w, c = img.shape                             # width and height of the img
                cx, cy = int(lm.x*w), int(lm.y*h)               # convert decimal values to pixel values

                # print(id, cx, cy)
                lmksList.append([id, cx, cy])
                # if id == 4:                                   # id = location of the landmark
                    # draw the circle
                if draw:
                    cv2.circle(img,                             # draw circle for easier figuring the in-consider landmark
                               (cx, cy),                        # coordinate of the landmark in pixels
                               6,                               # radius of circle
                               (50, 255, 50),                   # color value (BGR)
                               cv2.FILLED)                      # fill the drawn circle

        return lmksList


def main():
    pTime = 0                                                   # current time
    cTime = 0                                                   # previous time

    # create video object
    cap = cv2.VideoCapture(0)                                   # use webcam 0
    detector = handDetector()

    while True:
        success, img = cap.read()                               # get the frames
        img = cv2.flip(img, 1)                                  # flip the image from front camera

        img = detector.findHands(img)                           # send the image to the detector
        lmksList = detector.findPosition(img)

        if len(lmksList) != 0:                                  # check if nothing is found
            print(lmksList[0])                                  # print the value of the list's index

        # see fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img,                                        # put it into img
                    str(int(fps)),                              # str return decimal value, convert to int
                    (10, 60),                                   # set position
                    cv2.FONT_HERSHEY_SIMPLEX,                   # set Font
                    2,                                          # font scale
                    (0, 0, 255),                                # color value (BGR) (let's take red)
                    3)                                          # text thickness

        # let's see the webcam
        cv2.imshow("Image", img)
        cv2.waitKey(1)                                          # waitKey is used to display a frame for 1ms, after which display will be automatically closed


if __name__ == "__main__":
    main()

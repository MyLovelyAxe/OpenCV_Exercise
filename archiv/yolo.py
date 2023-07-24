import numpy as np
import cv2

cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
cap.set(3,640) # set height, whose id is 3 in setting
cap.set(4,480) # set width, whose id is 4 in setting
cap.set(10,20) # set brightness, whose id is 10 in setting

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as F:
    classesNames = F.read().rstrip('\n').split('\n')
print(classesNames)
print(f'We have {len(classNames)} classes')

while True:
    success, img = cap.read() # read frames one by one, return img and whether it is successful
    cv2.imshow('Video', img) # show the current frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for specific keyboard 'q' to close the window
        break
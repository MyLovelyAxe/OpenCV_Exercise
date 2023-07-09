import cv2
import numpy as np
from utils import plot_in_one_window

def get_fixed_size():
    wImg,hImg = 320,240
    return wImg,hImg

def config_webcom():
    cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
    wImg,hImg = get_fixed_size()
    cap.set(3,wImg) # set height, whose id is 3 in setting
    cap.set(4,hImg) # set width, whose id is 4 in setting
    cap.set(10,20) # set brightness, whose id is 10 in setting
    return cap

def main():
    cap = config_webcom()
    FaceDetector = cv2.CascadeClassifier('archiv/haarcascade_frontalface_default.xml')
    
    while True:
        # wait for specific keyboard 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # read frames one by one, return img and whether it is successful
        _, img = cap.read()
        wImg,hImg = get_fixed_size()
        img = cv2.resize(img,(wImg,hImg))
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBox = img.copy()
        NumPlates = FaceDetector.detectMultiScale(image=imgGray,scaleFactor=1.1,minNeighbors=4)
        if not len(NumPlates) == 0:
            for (x,y,w,h) in NumPlates:
                cv2.rectangle(img=imgBox,
                            pt1=(x,y),
                            pt2=(x+w,y+h),
                            color=(255,0,0),
                            thickness=2)
        else:
            print('No face detected')
        imgRes = plot_in_one_window([[img,imgBox]])
        cv2.imshow('Result', imgRes)

if __name__ == '__main__':

    main()

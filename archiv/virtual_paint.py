import cv2
import numpy as np
from utils import plot_in_one_window

def config_webcom():
    cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
    cap.set(3,640) # set height, whose id is 3 in setting
    cap.set(4,480) # set width, whose id is 4 in setting
    cap.set(10,20) # set brightness, whose id is 10 in setting
    return cap

def getHSV(img):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return imgHSV

def createTrackBar(name):
    cv2.namedWindow(name)
    cv2.resizeWindow(name,640,640)
    def do_nothing():
        pass
    # args:
    #  1. track bar name
    #  2. on which window
    #  3. starting value of bar
    #  4. maximum of bar
    #  5. a function to be called every time when track bar changes
    cv2.createTrackbar('Hue min',name,0,179,do_nothing)
    cv2.createTrackbar('Hue max',name,179,179,do_nothing)
    cv2.createTrackbar('Sat min',name,0,255,do_nothing)
    cv2.createTrackbar('Sat max',name,255,255,do_nothing)
    cv2.createTrackbar('Val min',name,0,255,do_nothing)
    cv2.createTrackbar('Val max',name,255,255,do_nothing)

def getMaskedImage(imgHSV):
    hue_min = cv2.getTrackbarPos('Hue min', 'TrackBars')
    hue_max = cv2.getTrackbarPos('Hue max', 'TrackBars')
    sat_min = cv2.getTrackbarPos('Sat min', 'TrackBars')
    sat_max = cv2.getTrackbarPos('Sat max', 'TrackBars')
    val_min = cv2.getTrackbarPos('Val min', 'TrackBars')
    val_max = cv2.getTrackbarPos('Val max', 'TrackBars')
    ### make a mask to extract the specific color
    # hue: color scale
    # saturation: degree of color, e.g. how red is it?
    # value: the illumination of color
    lower_limit = np.array([hue_min,sat_min,val_min])
    upper_limit = np.array([hue_max,sat_max,val_max])
    Mask = cv2.inRange(imgHSV,lower_limit,upper_limit)
    ### apply bitwise 'and' operation to extract from original image
    # args:
    #  1. original image
    #  2. output image
    #  3. mask
    # Attention:
    #  cv2.bitwise_and only take mask with 2D
    MaskedImage = cv2.bitwise_and(img,img,mask=Mask)
    return MaskedImage

if __name__ == '__main__':
    cap = config_webcom()
    createTrackBar('TrackBars')
    while True:
        # wait for specific keyboard 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # read frames one by one, return img and whether it is successful
        success, img = cap.read()
        imgHSV = getHSV(img)
        MaskedImage = getMaskedImage(imgHSV)
        show_img = plot_in_one_window([[imgHSV,MaskedImage]])
        cv2.imshow('Video', show_img)

import argparse
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

def getContours(Mask):
    """
    get contours of object
    """
    contours,_ = cv2.findContours(image=Mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    print(f'how many contours: {len(contours)}')
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        # calculate perimeter
        peri = cv2.arcLength(curve=cnt,closed=True)
        # get approximated polygone
        approx = cv2.approxPolyDP(curve=cnt,epsilon=0.02*peri,closed=True)
        # get bounding box
        x,y,w,h = cv2.boundingRect(array=approx)
    return x,y,w,h

def createTrackBar(name):
    cv2.namedWindow(name)
    cv2.resizeWindow(name,640,640)
    def do_nothing():
        pass
        # create Trackbars
    cv2.createTrackbar('Hue min',name,0,179,do_nothing)
    cv2.createTrackbar('Hue max',name,179,179,do_nothing)
    cv2.createTrackbar('Sat min',name,0,255,do_nothing)
    cv2.createTrackbar('Sat max',name,255,255,do_nothing)
    cv2.createTrackbar('Val min',name,0,255,do_nothing)
    cv2.createTrackbar('Val max',name,255,255,do_nothing)

def getMaskedImage(imgHSV,ori_img,mask_range=None):
    if mask_range is None:
        hue_min = cv2.getTrackbarPos('Hue min', 'TrackBars')
        hue_max = cv2.getTrackbarPos('Hue max', 'TrackBars')
        sat_min = cv2.getTrackbarPos('Sat min', 'TrackBars')
        sat_max = cv2.getTrackbarPos('Sat max', 'TrackBars')
        val_min = cv2.getTrackbarPos('Val min', 'TrackBars')
        val_max = cv2.getTrackbarPos('Val max', 'TrackBars')
        # make a mask to extract the specific color
        lower_limit = np.array([hue_min,sat_min,val_min])
        upper_limit = np.array([hue_max,sat_max,val_max])
    else:
        lower_limit,upper_limit = mask_range
    Mask = cv2.inRange(imgHSV,lower_limit,upper_limit)
    # apply bitwise 'and' operation to extract from original image
    # Attention:
    #  cv2.bitwise_and only take mask with 2D
    MaskedImage = cv2.bitwise_and(ori_img,ori_img,mask=Mask)
    x,y,w,h = getContours(Mask)
    return MaskedImage,[x,y,w,h]

def getColor():
    """
    run function testColor() and type in the found values
    """
    lower_limit = np.array([0,128,103]) # hue_min,sat_min,val_min
    upper_limit = np.array([8,178,198]) # hue_max,sat_max,val_max
    return lower_limit,upper_limit

def main(task:str):
    """
    tweak tracking bars to find the range of desired colors
    """
    cap = config_webcom()
    points = []
    if task == 'test_color':
        createTrackBar('TrackBars')
    while True:
        # wait for specific keyboard 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # read frames one by one, return img and whether it is successful
        _, img = cap.read()
        imgHSV = getHSV(img)
        if task == 'test_color':
            MaskedImage,[x,y,w,h] = getMaskedImage(imgHSV,img)
        elif task == 'air_painting':
            MaskedImage,[x,y,w,h] = getMaskedImage(imgHSV,img,getColor())
            if x != 0 and y!= 0:
                pen_top = [x+w//2,y]
                points.append(pen_top)
        imgContour = img.copy()
        for point in points:
            cv2.circle(imgContour,(point[0],point[1]),10,(0,0,255),cv2.FILLED)
        show_img = plot_in_one_window([[MaskedImage,imgContour]])
        cv2.imshow('Video', show_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Air painting')
    parser.add_argument('--Task',type=str,default='air_painting',choices=['test_color','air_painting'])
    args = parser.parse_args([])
    main(args.Task)

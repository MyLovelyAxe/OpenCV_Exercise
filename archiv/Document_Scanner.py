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

def preprocess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,ksize=(5,5),sigmaX=1)
    imgCanny = cv2.Canny(imgBlur,threshold1=200,threshold2=200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold = cv2.erode(imgDial,kernel,iterations=1)
    return imgThreshold

def getContours(imgThres):
    """
    get contours of the biggest rectangle-object
    """
    maxArea = 0
    biggest = np.array([])
    contours,_ = cv2.findContours(image=imgThres,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    # print(f'how many contours: {len(contours)}')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            peri = cv2.arcLength(curve=cnt,closed=True)
            # get approximated polygone
            approx = cv2.approxPolyDP(curve=cnt,epsilon=0.02*peri,closed=True)
            # if the number of approximated points is 4, i.e. rectangle
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reorder(points):
    """
    calculate summation of x+y for all 4 corners from points
    the smallest summation gives position of left-upper corner
    the largest summation gives position of right-lower corner
    """
    points = points.reshape((4,2))
    reordered_points = np.zeros((4,1,2),np.float32)
    # left-upper corner & right_lower corner
    add = np.sum(points,axis=1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    # left-right corner & left_lower corner
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    return reordered_points

def getWarp(imgWarp,wImg,hImg,biggest):
    """
    biggest points have shape [4,1,2] = [num_points,1_extra_dim,2_x_and_y]
    before warpping, biggest should squeeze dim=1 and be re-ordered
    """
    pts1 = reorder(biggest)
    pts2 = np.float32([[0,0],[wImg,0],[0,hImg],[wImg,hImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(imgWarp, matrix, (wImg, hImg))
    return imgOutput

def main():
    cap = config_webcom()
    while True:
        # wait for specific keyboard 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # read frames one by one, return img and whether it is successful
        _, img = cap.read()
        wImg,hImg = get_fixed_size()
        img = cv2.resize(img,(wImg,hImg))
        # image after
        imgThres = preprocess(img)
        imgContour = img.copy()
        imgWarp = img.copy()
        biggest = getContours(imgThres)
        # if there are rectangle detected, show the warpped result
        if not len(biggest) == 0:
            cv2.drawContours(image=imgContour,
                            contours=biggest,
                            contourIdx=-1,
                            color=(255,125,0),
                            thickness=5)
            imgWarp = getWarp(imgWarp,wImg,hImg,biggest)
            imgRes = plot_in_one_window([[imgThres,imgContour,imgWarp]])
        else:
            print('no biggest contours detected')
            imgRes = plot_in_one_window([[imgThres,imgContour]])
        cv2.imshow('Result', imgRes)

if __name__ == '__main__':

    main()

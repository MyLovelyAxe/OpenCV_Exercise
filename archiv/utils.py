import cv2
import numpy as np
import os

print(f'current path: {os.getcwd()}')

def plot_in_one_window(img_lst:list):
    """
    img_lst should be a list containing multiple lists:
        e.g. img_lst = [[img1-1,img1-2],[img2-1,img2-2],[img3-1,img,3-2]]
        this will show one window containing 6 plots deployed as 3x2 subplots
    """
    img_shapes = None
    shape_lst = []
    for line in img_lst:
        for img in line:
            print(img)
            shape_lst.append(len(img.shape))
    if len(set(shape_lst)) == 1:
        if set(shape_lst).pop() == 2:
            img_shapes = 2
        elif set(shape_lst).pop() == 3:
            img_shapes = 3
        del shape_lst
        row = np.NAN
        col = np.NAN
        for line in img_lst:
            for img in line:
                row = np.hstack((row,img))
            col = np.vstack((col,row))
    else:
        del shape_lst
        row = np.NAN
        col = np.NAN
        for line in img_lst:
            for img in line:
                if len(img.shape) == 2:
                    # when counter with 2D image, convert it into 3D
                    # to be compatible with other plots
                    img = np.stack((img,)*3, axis=-1)
                row = np.hstack((row,img))
            col = np.vstack((col,row))
    final_image = col
    del row,col
    return final_image

image = cv2.imread('geometry.jpg')
print(image.shape)
# final_image = plot_in_one_window(img_lst=[[image],[image],[image]])

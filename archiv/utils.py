import cv2
import numpy as np


def plot_in_one_window(img_lst:list):
    """
    Plot multiple images into one window

    Arguments:
        img_lst:
            list containing multiple lists:
            e.g. img_lst = [[img1-1,img1-2],[img2-1,img2-2],[img3-1,img,3-2]]
            this will show one window containing 6 plots deployed as 3x2 subplots
    """
    ### prepare
    num_row = len(img_lst)
    num_col = len(img_lst[0])
    h,w = img_lst[0][0].shape[0],img_lst[0][0].shape[1]
    same = True # label for whether all images have same shape
    final_image = None
    ### check shapes of all images
    shape_lst = []
    for line in img_lst:
        for img in line:
            shape_lst.append(len(img.shape))
    ### all images are of same shape
    if len(set(shape_lst)) == 1:
        # 2D images
        if set(shape_lst).pop() == 2:
            final_image = np.zeros((h*num_row,w*num_col),np.uint8)
        # 3D images
        elif set(shape_lst).pop() == 3:
            final_image = np.zeros((h*num_row,w*num_col,3),np.uint8)
    ### there are 2D and 3D images in proposal
    else:
        final_image = np.zeros((h*num_row,w*num_col,3),np.uint8)
        same = False
    del shape_lst
    for row,line in enumerate(img_lst):
        for col,img in enumerate(line):
            if len(img.shape) == 2 and same is False:
                # when counter with 2D image, convert it into 3D
                # to be compatible with other plots
                img = np.stack((img,)*3, axis=-1)
            final_image[h*row:h*(row+1),w*col:w*(col+1)] = img
    return final_image

def test_img(img_path:str,scale:int):
    ### get original image
    image = cv2.imread(img_path)
    H,W,C = image.shape
    image = cv2.resize(image,(int(W/scale),int(H/scale)))
    print(f'original image shape: {image.shape}')
    ### some variants
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f'imgGray shape: {imgGray.shape}')
    imgBlur = cv2.GaussianBlur(image, ksize=[5,5], sigmaX=1)
    print(f'imgBlur shape: {imgBlur.shape}')
    ### join all images into one window
    FinalImage = plot_in_one_window(img_lst=[[image,imgGray,imgGray],
                                              [imgGray,imgBlur,image]])
    print(f'imgBlur shape: {FinalImage.shape}')
    ### show window
    cv2.imshow('test',FinalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_img(img_path='redhood.jpg',scale=5)

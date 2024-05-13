import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 

class DepthMap: 
    def __init__(self,showImages, l, r): 
        # Load Images 
        root = os.getcwd()
        # imgLeftPath = os.path.join(root, 'koon_calibration/left/img45.png')
        # imgRightPath = os.path.join(root,'koon_calibration/right/img45.png')
        # self.imgLeft = cv.imread(imgLeftPath,cv.IMREAD_GRAYSCALE) 
        # self.imgRight = cv.imread(imgRightPath,cv.IMREAD_GRAYSCALE) 
        self.imgLeft = cv.cvtColor(l, cv.COLOR_BGR2GRAY)
        self.imgRight = cv.cvtColor(r, cv.COLOR_BGR2GRAY)

        if showImages: 
            plt.figure() 
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show() 

    def computeDepthMapBM(self):
        nDispFactor = 12 # adjust this 
        stereo = cv.StereoBM.create(numDisparities=16*nDispFactor, blockSize=21)
        disparity = stereo.compute(self.imgLeft,self.imgRight)
        plt.imshow(disparity,'gray')
        plt.show()

    def computeDepthMapSGBM(self): 
        window_size = 7
        min_disp = 16
        nDispFactor = 14 # adjust this (14 is good)
        num_disp = 16 * nDispFactor - min_disp

        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    blockSize=window_size,
                                    P1=8 * 3 * window_size ** 2,
                                    P2=32 * 3 * window_size ** 2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=15,
                                    speckleWindowSize=0,
                                    speckleRange=2,
                                    preFilterCap=63,
                                    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

        # Compute disparity map
        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 255.0

        # Normalize the disparity map for better visualization
        disparity_normalized = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # Apply a colormap to the normalized disparity map
        disparity_colored = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)

        # Display the colored disparity map
        cv.imshow("disp", disparity_colored)
        # plt.imshow(disparity_colored)
        cv.waitKey(1)



def demoViewPics():
    # See pictures 
    dp = DepthMap(showImages=True) 

def demoStereoBM(): 
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()

def demoStereoSGBM(): 
    left = cv.VideoCapture('test_video_left.mp4')
    right = cv.VideoCapture('test_video_right.mp4')
    while True:
        _, l = left.read()
        _, r = right.read()
        l = cv.resize(l, (636, 474))
        r = cv.resize(r, (636, 474))
        dp = DepthMap(False, l, r)
        dp.computeDepthMapSGBM()

if __name__ == '__main__': 
    # demoViewPics()
    # demoStereoBM()
    demoStereoSGBM() 
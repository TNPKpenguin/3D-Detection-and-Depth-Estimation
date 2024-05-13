import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 

class DepthMap: 
    def __init__(self,showImages, l, r, d): 
        # Load Images 
        root = os.getcwd()
        # imgLeftPath = os.path.join(root, 'koon_calibration/left/img45.png')
        # imgRightPath = os.path.join(root,'koon_calibration/right/img45.png')
        # self.imgLeft = cv.imread(imgLeftPath,cv.IMREAD_GRAYSCALE) 
        # self.imgRight = cv.imread(imgRightPath,cv.IMREAD_GRAYSCALE) 
        self.imgDetected = d.copy()
        self.l = l 
        self.r = r
        self.imgLeft = cv.cvtColor(l, cv.COLOR_BGR2GRAY)
        self.imgRight = cv.cvtColor(r, cv.COLOR_BGR2GRAY)

        if showImages: 
            plt.figure() 
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show() 

    def find_depth(self, right_point, left_point, frame_right, frame_left, baseline,f, alpha):

        # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
        height_right, width_right, depth_right = frame_right.shape
        height_left, width_left, depth_left = frame_left.shape

        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

        else:
            print('Left and right camera frames do not have the same pixel width')

        x_right = right_point[0]
        x_left = left_point[0]

        # CALCULATE THE DISPARITY:
        disparity = x_left-x_right      #Displacement between left and right frames [pixels]

        # CALCULATE DEPTH z:
        zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

        return zDepth


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

        # Convert image to HSV
        hsv_img = cv.cvtColor(self.imgDetected, cv.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv_img, lower_green, upper_green)

        # Find contours of green regions
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blended_image = disparity_colored.copy()
        # Draw bounding boxes around green regions
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            depth =self.find_depth(x,y, self.l, self.r, 9, 8, 56.6)
            print(depth)
            cv.rectangle(disparity_colored, (x, y), (x + w, y + h-20), (0, 0, 255), 3)
            blended_image[y:y+h, x:x+w] = self.imgDetected[y:y+h, x:x+w]
            # depth_mean = np.mean(disparity[y:y+h, x:x+w])
            # # Display depth text
            # cv.putText(disparity_colored, f"{depth_mean:.2f}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # Blended Image
        # alpha = 1
        # blended_image = cv.addWeighted(disparity_colored, alpha, self.imgDetected, 1 - alpha, 0, dtype=cv.CV_32F)

        # Display the colored disparity map with green regions highlighted
        cv.imshow("detect", self.imgDetected)
        cv.imshow("disp", disparity_colored)
        cv.imshow("blend", blended_image)

        cv.waitKey(1)

    

def demoViewPics():
    # See pictures 
    dp = DepthMap(showImages=True) 

def demoStereoBM(): 
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()



def demoStereoSGBM(): 
    cap = cv.VideoCapture('test_video_left.mp4')
    left = cv.VideoCapture('video_left.mp4')
    right = cv.VideoCapture('video_right.mp4')
    while True:
        _, frame = cap.read()
        _, l = left.read()
        _, r = right.read()
        frame = cv.resize(frame, (636, 474))
        l = cv.resize(l, (636, 474))
        r = cv.resize(r, (636, 474))
        dp = DepthMap(False, l, r, frame)
        dp.computeDepthMapSGBM()
        
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break 

if __name__ == '__main__': 
    # demoViewPics()
    # demoStereoBM()
    demoStereoSGBM() 
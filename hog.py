import cv2
import numpy as np
from imutils.object_detection import non_max_suppression 

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
        # for cnt in contours:
        #     x, y, w, h = cv.boundingRect(cnt)
        #     cv.rectangle(disparity_colored, (x, y), (x + w, y + h-20), (0, 0, 255), 3)
        #     blended_image[y:y+h, x:x+w] = self.imgDetected[y:y+h, x:x+w]
            # depth_mean = np.mean(disparity[y:y+h, x:x+w])
            # # Display depth text
            # cv.putText(disparity_colored, f"{depth_mean:.2f}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # Blended Image
        # alpha = 1
        # blended_image = cv.addWeighted(disparity_colored, alpha, self.imgDetected, 1 - alpha, 0, dtype=cv.CV_32F)

        # Display the colored disparity map with green regions highlighted
        # cv.imshow("detect", self.imgDetected)
        # cv.imshow("disp", disparity_colored)
        # cv.imshow("blend", blended_image)

        return disparity_colored

    

def demoViewPics():
    # See pictures 
    dp = DepthMap(showImages=True) 

def demoStereoBM(): 
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()

thresh = 0.5
# image = cv2.imread('pokemon.png')
# template = cv2.imread('pokemon_template.png')

# cap_left = cv2.VideoCapture("video_left.mp4")
# cap_right = cv2.VideoCapture("video_right.mp4")


def demoStereoSGBM(): 
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())    
    cap = cv.VideoCapture('test_video_left.mp4')
    left = cv.VideoCapture('video_left.mp4')
    right = cv.VideoCapture('video_right.mp4')
    while True:
        template = cv2.imread('pokemon_template.png')
        _, frame = cap.read()
        _, l = left.read()
        _, r = right.read()

        
        frame = cv.resize(frame, (636, 474))
        l = cv.resize(l, (636, 474))
        r = cv.resize(r, (636, 474))
        dp = DepthMap(False, l, r, frame)
        disp = dp.computeDepthMapSGBM()

        locations, confidence = hog.detectMultiScale(l)
        for (x, y, w, h) in locations:
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 5)
        

        cv2.imshow("template matching disparity", disp)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break 
if __name__ == '__main__': 
    # demoViewPics()
    # demoStereoBM()
    demoStereoSGBM() 
# while True:
#     _, l = cap_left.read()
#     _, r = cap_right.read()
#     img_gray = cv2.cvtColor(l,  cv2.COLOR_BGR2GRAY) 
#     temp_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) 

#     W, H = template.shape[:2] 

#     template_height, template_width = template.shape[:2]

#     match = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
#     (y_points, x_points) = np.where(match >= thresh) 
    
#     boxes = list() 
    
#     for (x, y) in zip(x_points, y_points): 
#         boxes.append((x, y, x + W, y + H)) 


#     boxes = non_max_suppression(np.array(boxes))

#     for (x1, y1, x2, y2) in boxes: 
#         cv2.rectangle(image, (x1, y1), (x2, y2), 
#         (255, 0, 0),3) 
        
#     cv2.imshow("Template" ,template) 
#     cv2.imshow("After NMS", image) 
#     cv2.waitKey(0)
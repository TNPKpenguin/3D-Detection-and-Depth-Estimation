# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# # left_image = cv.imread('koon_calibration/left/img5.png', cv.IMREAD_GRAYSCALE)
# # right_image = cv.imread('koon_calibration/right/img5.png', cv.IMREAD_GRAYSCALE)

# left_image = cv.imread('stereo-reconstruction/bike/im0.png', cv.IMREAD_GRAYSCALE)
# right_image = cv.imread('stereo-reconstruction/bike/im1.png', cv.IMREAD_GRAYSCALE)

# stereo = cv.StereoBM_create(numDisparities=16, blockSize=17)
# depth = stereo.compute(left_image, right_image)

# # Normalize the depth map for visualization
# depth_map_normalized = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

# # Apply colormap to the depth map
# depth_colormap = cv.applyColorMap(depth_map_normalized, cv.COLORMAP_JET)

# cv.imshow("Left", left_image)
# cv.imshow("Right", right_image)
# cv.imshow("Depth", depth_colormap)
# cv.waitKey(0)
# cv.destroyAllWindows()

# ----------------------------------------------------------------------------------#
# !/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import yaml

def config_info(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    Dic = {}
    for l in Lines:
        a = l.split("=")
        if(a[1][-1] == "\n"):
            try:
                Dic[a[0]] = int(a[1][:-1])
            except:
                Dic[a[0]] = a[1][:-1]
        else:
            try:
                Dic[a[0]] = int(a[1])
            except:
                Dic[a[0]] = a[1]
    Dic["cam0"] = Dic["cam0"].split(" ")
    Dic["cam0"][0] = Dic["cam0"][0][1:]
    Dic["cam0"][2] = Dic["cam0"][2][:-1]
    Dic["cam0"][5] = Dic["cam0"][5][:-1]
    Dic["cam0"][-1] = Dic["cam0"][-1][:-1]
    Dic["cam1"] = Dic["cam1"].split(" ")
    Dic["cam1"][0] = Dic["cam1"][0][1:]
    Dic["cam1"][2] = Dic["cam1"][2][:-1]
    Dic["cam1"][5] = Dic["cam1"][5][:-1]
    Dic["cam1"][-1] = Dic["cam1"][-1][:-1]
    for i in range(len(Dic["cam0"])):
        Dic["cam0"][i] = float(Dic["cam0"][i])
        Dic["cam1"][i] = float(Dic["cam1"][i])
    Dic["cam0"] = np.array(Dic["cam0"]).reshape(3,3)
    Dic["cam1"] = np.array(Dic["cam1"]).reshape(3,3)
    max_disparity = Dic["vmax"]
    min_disparity = Dic["vmin"]
    num_disparities = max_disparity - min_disparity
    # print(num_disparities)
    window_size = 7 #31
    k = Dic["cam0"]
    distortion = np.zeros((5,1)).astype(np.float32)
    T = np.zeros((3,1))
    T[0,0] = Dic["baseline"]
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(k, distortion, k, distortion, (Dic["height"], Dic["width"]), np.identity(3), T)
    return Dic, k, Q, max_disparity, min_disparity, num_disparities, window_size

def find_disparity(image1, image2, path):
    Dic, k, Q, max_disparity, min_disparity, num_disparities, window_size = config_info(path)
    stereo = cv.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, preFilterCap=1, blockSize=2, uniquenessRatio=2, speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1, P1=8*3*window_size**2, P2=32*3*window_size**2, mode=4)
    imgL = cv.cvtColor(image1.copy(), cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(image1.copy(), cv.COLOR_BGR2GRAY)
    disparity = stereo.compute(imgL, imgR).astype(np.float32)
    disparity = cv.medianBlur(disparity, 5)  # Apply median filter to remove noise
    return disparity, Q

def disparity_to_depth(disparity, Q):
    depth_map = cv.reprojectImageTo3D(disparity, Q)[:, :, 2]
    return depth_map

def depth_to_color(depth_map):
    # Define color map ranges (adjust these according to your depth values)
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    
    # Normalize depth map
    depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    
    # Apply colormap
    depth_map_color = cv.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv.COLORMAP_JET)
    return depth_map_color
    
def detect(image):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    image = cv.resize(image, (640, 480))

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)


    # for (x, y, w, h) in rects:
    #     box_pos.append([x, y, w, h])
        # cv.rectangle(image, (x, y), (x + w+30, y + h), (0, 255, 0), 2)
    return rects

cap = cv.VideoCapture('test_video_left.mp4')
left = cv.VideoCapture('video_left.mp4')
right = cv.VideoCapture('video_right.mp4')
# if __name__ == "__main__":
#         while True:
#             _, frame = cap.read()
#             _, l = left.read()
#             _, r = right.read()
#             image_1_path = l.copy()
#             image_2_path = r.copy()
#             # image_1_path = f"pic-zip\\stereoLeft\\img{i*2}.png"
#             # image_2_path = f"pic-zip\\stereoRight\img{i*2}.png"
#             calibration_info_path = "C:\\Users\\LENOVO\\Desktop\\HCU\\HCU3_2\\Computer Vision\\camera_calibration\\stereo-reconstruction\\test\\calibration_info.txt"

#             disparity, Q = find_disparity(image_1_path, image_2_path, calibration_info_path)
#             depth_map = disparity_to_depth(disparity, Q)

#             image1 = frame.copy()
#             depth_map_resized = cv.resize(depth_map, (image1.shape[1], image1.shape[0]))

#             depth_map_color = depth_to_color(depth_map_resized)

#             image1_float = image1.astype(np.float32) / 255.0
#             alpha = 0.1
#             blended_image = cv.addWeighted(depth_map_color, alpha, image1_float, 1 - alpha, 0, dtype=cv.CV_32F)
#             blended_image = np.clip(blended_image * 255.0, 0, 255).astype(np.uint8)

#             # box_pos = detect(image1)

#             # for (x, y, w, h) in box_pos:
#             #     print(x, y, w, h)
#             #     cv.rectangle(blended_image, (x, y), (x + w+30, y + h), (0, 255, 0), 2)

#             # Display depth map
#             cv.imshow("depth map", blended_image)
#             cv.waitKey(1)
#             if cv.waitKey(1) & 0xFF == ord('q'):  # Change the waitKey to 0 if you want to wait until a key is pressed to continue
#                 break
if __name__ == "__main__":
    for i in range(1, 51):
        image_1_path = f"koon_calibration\\left\\img{i*2}.png"
        image_2_path = f"koon_calibration\\right\\img{i*2}.png"
        calibration_info_path = "C:\\Users\\LENOVO\\Desktop\\HCU\\HCU3_2\\Computer Vision\\camera_calibration\\stereo-reconstruction\\test\\calibration_info.txt"

        image1 = cv.imread(image_1_path)
        image2 = cv.imread(image_2_path)
        disparity, Q = find_disparity(image1, image2, calibration_info_path)
        depth_map = disparity_to_depth(disparity, Q)

        
        depth_map_resized = cv.resize(depth_map, (image1.shape[1], image1.shape[0]))
        depth_map_normalized = cv.normalize(depth_map_resized, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        depth_map_color = cv.cvtColor(depth_map_normalized, cv.COLOR_GRAY2BGR)
        depth_map_rgb = cv.cvtColor(depth_map_color, cv.COLOR_BGR2RGB)
        print(depth_map.shape)
        image1_float = image1.astype(np.float32) / 255.0
        alpha = 0.5
        blended_image = cv.addWeighted(image1_float, alpha, depth_map_rgb, 1 - alpha, 0, dtype=cv.CV_32F)
        disparity_normalized = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        disparity_colored = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)
        blended_image = np.clip(blended_image * 255.0, 0, 255).astype(np.uint8)
        
        # Display depth map
        cv.imshow("depth map", disparity_colored)
        cv.waitKey(3)

#-----------------------------------------------------#

# import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv

# def config_info():
#     # Input your configuration parameters here
#     cam0 = [[670.79395677, 0, 305.57472741], [0, 671.47436598, 233.18172062], [0, 0, 1]]
#     cam1 = [[670.88630708, 0 302.09869763], [0, 670.88364713, 274.56664311], [0, 0, 1]]
    
#     vmax = 256  # Maximum disparity
#     vmin = 0    # Minimum disparity
#     baseline = 0.1  # Baseline (distance between the two cameras)
#     height = 480    # Image height
#     width = 640     # Image width

#     return cam0, cam1, vmax, vmin, baseline, height, width

# def find_disparity(image1, image2, cam0, cam1, vmax, vmin, baseline, height, width):
#     # Stereo calibration
#     stereo = cv.StereoSGBM_create(minDisparity=vmin, numDisparities=vmax-vmin, preFilterCap=1, blockSize=5, uniquenessRatio=2, speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1, P1=8*3*5**2, P2=32*3*5**2, mode=4)

#     # Reading images
#     imgL = cv.imread(image1, 0)
#     imgR = cv.imread(image2, 0)

#     # Computing disparity map
#     disparity = stereo.compute(imgL, imgR).astype(np.float32)
#     disparity = cv.medianBlur(disparity, 5)  # Apply median filter to remove noise

#     # Stereo rectification
#     k = cam0
#     distortion = np.zeros((5,1)).astype(np.float32)
#     T = np.zeros((3,1))
#     T[0,0] = baseline
#     R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(k, distortion, k, distortion, (height, width), np.identity(3), T)

#     return disparity, Q

# def disparity_to_depth(disparity, Q):
#     depth_map = cv.reprojectImageTo3D(disparity, Q)[:, :, 2]
#     return depth_map

# if __name__ == "__main__":
#     cam0, cam1, vmax, vmin, baseline, height, width = config_info()
    
#     # Image paths
#     image1_path = "C:\\Users\\LENOVO\\Desktop\\HCU\\HCU3_2\\Computer Vision\\camera_calibration\\stereo-reconstruction\\test\\im0.png"
#     image2_path = "C:\\Users\\LENOVO\\Desktop\\HCU\\HCU3_2\\Computer Vision\\camera_calibration\\stereo-reconstruction\\test\\im1.png"
    
#     disparity, Q = find_disparity(image1_path, image2_path, cam0, cam1, vmax, vmin, baseline, height, width)
#     depth_map = disparity_to_depth(disparity, Q)

#     # Display depth map
#     plt.imshow(depth_map, cmap='jet')
#     plt.colorbar()
#     plt.show()

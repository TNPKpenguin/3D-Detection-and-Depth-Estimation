import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import yaml
import time 

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
    print(num_disparities)
    window_size = 31
    k = Dic["cam0"]
    distortion = np.zeros((5,1)).astype(np.float32)
    T = np.zeros((3,1))
    T[0,0] = Dic["baseline"]
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(k, distortion, k, distortion, (Dic["height"], Dic["width"]), np.identity(3), T)
    return Dic, k, Q, max_disparity, min_disparity, num_disparities, window_size

def find_disparity(image1, image2, path):
    Dic, k, Q, max_disparity, min_disparity, num_disparities, window_size = config_info(path)
    stereo = cv.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, preFilterCap=1, blockSize=5, uniquenessRatio=2, speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1, P1=8*3*window_size**2, P2=32*3*window_size**2, mode=4)
    disparity = stereo.compute(image1, image2).astype(np.float32)
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

def generate_depth(left, right):
    image_1 = left
    image_2 = right
    calibration_info_path = "C:\\Users\\LENOVO\\Desktop\\HCU\\HCU3_2\\Computer Vision\\camera_calibration\\stereo-reconstruction\\test\\calibration_info.txt"

    disparity, Q = find_disparity(image_1, image_2, calibration_info_path)
    depth_map = disparity_to_depth(disparity, Q)

    depth_map_resized = cv.resize(depth_map, (image_1.shape[1], image_1.shape[0]))

    depth_map_color = depth_to_color(depth_map_resized)
    image1_float = image_1.astype(np.float32) / 255.0
    alpha = 0.9
    blended_image = cv.addWeighted(image1_float, alpha, depth_map_color, 1 - alpha, 0, dtype=cv.CV_32F)        
    blended_image = np.clip(blended_image * 255.0, 0, 255).astype(np.uint8)
   
    return blended_image

cap_left = cv.VideoCapture("video_left.mp4")
cap_right = cv.VideoCapture("video_right.mp4")

# cap_left = cv.VideoCapture(0)
# cap_right = cv.VideoCapture(1)

start_time = time.time()
while True:
    ret, frame_left = cap_left.read()
    ret, frame_right = cap_right.read()
    frame_left = cv.resize(frame_left, (636, 474))
    frame_right = cv.resize(frame_right, (636, 474))
    if not ret:
        break
    depth_map = generate_depth(frame_left, frame_right)
    print(depth_map.shape)
    cv.imshow("original", depth_map)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
end_time = time.time()

print("Time usage :", end_time - start_time)
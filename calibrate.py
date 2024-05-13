import cv2
import numpy as np
import os

# Define the calibration pattern (e.g., a checkerboard)
pattern_size = (9, 6)  # Number of inner corners in the calibration pattern
square_size = 0.0245   # Size of each square in meters (adjust according to your calibration pattern)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points in real world space
left_imgpoints = []  # 2D points in left image plane
right_imgpoints = []  # 2D points in right image plane

# Load left and right calibration images
left_image_folder = 'new_image2/left/'
right_image_folder = 'new_image2/right/'

# Load left images
left_images = os.listdir(left_image_folder)
left_images = [os.path.join(left_image_folder, img) for img in left_images if img.endswith('.png')]

for img_path in left_images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(np.zeros((np.prod(pattern_size), 3), np.float32))
        objpoints[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        left_imgpoints.append(corners)

        # Calibrate left camera
        ret1, K1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera([objpoints[-1]], [left_imgpoints[-1]], gray.shape[::-1], None, None)

        # Extract intrinsic parameters for left camera
        fx_left = K1[0, 0]
        fy_left = K1[1, 1]
        cx_left = K1[0, 2]
        cy_left = K1[1, 2]

        print("Left camera intrinsic parameters:")
        print("fx:", fx_left)
        print("fy:", fy_left)
        print("cx:", cx_left)
        print("cy:", cy_left)
    else:
        print("Chessboard not found in:", img_path)

# Load right images
right_images = os.listdir(right_image_folder)
right_images = [os.path.join(right_image_folder, img) for img in right_images if img.endswith('.png')]

for img_path in right_images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add image points
    if ret:
        right_imgpoints.append(corners)

        # Calibrate right camera
        ret2, K2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, right_imgpoints, gray.shape[::-1], None, None)

        # Extract intrinsic parameters for right camera
        fx_right = K2[0, 0]
        fy_right = K2[1, 1]
        cx_right = K2[0, 2]
        cy_right = K2[1, 2]

        print("\nRight camera intrinsic parameters:")
        print("fx:", fx_right)
        print("fy:", fy_right)
        print("cx:", cx_right)
        print("cy:", cy_right)
    else:
        print("Chessboard not found in:", img_path)

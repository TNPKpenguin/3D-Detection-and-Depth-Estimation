import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images
try:
    img1 = cv2.imread('notebook/left/img2.png', 0)  # query image (left)
    img2 = cv2.imread('notebook/right/img2.png', 0)  # train image (right)
except Exception as e:
    print("Error loading images:", e)
    exit()

# Check if images are loaded properly
if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Initialize SIFT
try:
    sift = cv2.SIFT_create()
except cv2.error as e:
    print("Error initializing SIFT:", e)
    exit()

# Find keypoints and descriptors
try:
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
except cv2.error as e:
    print("Error detecting keypoints and descriptors:", e)
    exit()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Initialize FLANN matcher
try:
    flann = cv2.FlannBasedMatcher(index_params, search_params)
except cv2.error as e:
    print("Error initializing FLANN matcher:", e)
    exit()

# Match keypoints
try:
    matches = flann.knnMatch(des1, des2, k=2)
except cv2.error as e:
    print("Error matching keypoints:", e)
    exit()

# Apply Lowe's ratio test to select good matches
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.imshow(img_matches)
plt.show()

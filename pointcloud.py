import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the point cloud data from the text file
df = pd.read_csv("output/point_cloud_256.txt")

# Filter points with coordinates [0, 0, 0]
filtered_df = df[(df['x'] <= 250) & (df['y'] <= 250) & (df['z'] <= 250)]

# Extract x, y, z coordinates and RGB values
x = filtered_df['x']
y = filtered_df['y']
z = filtered_df['z']
r = filtered_df['r'] / 255.0  # Scale RGB values to range [0, 1]
g = filtered_df['g'] / 255.0
b = filtered_df['b'] / 255.0

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=list(zip(r, g, b)), marker='.')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization (Points at [0, 0, 0])')

# Show plot
plt.show()

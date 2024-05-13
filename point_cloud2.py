import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# อ่านข้อมูลจากไฟล์ text
df = pd.read_csv('output/point_cloud_256.txt')

# สร้างภาพ 3 มิติ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# พล็อตจุดและใช้ค่า z เป็นความลึก
ax.scatter(df['x'], df['y'], df['z'], c=df['z'], cmap='viridis', s=2)

# กำหนดชื่อแกน
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

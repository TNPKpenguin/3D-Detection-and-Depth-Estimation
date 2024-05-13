import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
from scipy import stats
import pymeshlab as ml

class ImageProcessor:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.array = np.array(self.image)
        self.info = f"Image Size: {self.array.shape}"

    def resize(self, width, height):
        self.image = self.image.resize((width, height))
        self.array = np.array(self.image)

    def get_array(self):
        return self.array

    def get_info(self):
        print(self.info)

def pre_process(image_path, height, width):
    image_processor = ImageProcessor(image_path)
    image_processor.resize(width, height)
    array = image_processor.get_array()

    print(f"Pre-processing for {image_path} complete.")
    print("Information:")
    image_processor.get_info()

    return array

def calculate_disparity(array_left, array_right, window_size, search_range):
    start_time = time.time()

    disp_matrix = []

    for row in range(len(array_left) - window_size):
        if row % 10 == 0:
            print(f"Disparity calculated for {row} rows.")

        disps = []

        for col1 in range(len(array_left[row]) - window_size):
            win1 = array_left[row:row + window_size, col1:col1 + window_size].flatten()

            if col1 < search_range:
                init = 0
            else:
                init = col1 - search_range

            sads = []

            for col2 in range(col1, init - 1, -1):
                win2 = array_right[row:row + window_size, col2:col2 + window_size].flatten()
                sad = np.sum(np.abs(np.subtract(win1, win2)))
                sads.append(sad)

            disparity = np.argmin(sads)
            disps.append(disparity)

        disp_matrix.append(disps)

    disp_matrix = np.array(disp_matrix)

    end_time = time.time()

    print("Disparity calculations complete.")
    print(f"Time elapsed during disparity calculations: {end_time - start_time}s")

    return disp_matrix

def post_process(disp_matrix):
    pp_disp = np.copy(disp_matrix)

    for x in range(pp_disp.shape[1]):
        for y in range(pp_disp.shape[0]):

            # MEAN
            avg_window = pp_disp[max(0, y - 7):min(pp_disp.shape[0], y + 8), max(0, x - 7):min(pp_disp.shape[1], x + 8)]
            if avg_window.size > 0:
                avg = np.mean(avg_window)
                if np.absolute(pp_disp[y, x] - avg) > 20:
                    pp_disp[y, x] = avg

            # MODE
            if x > 12 and x < (pp_disp.shape[0] - 12) and y > 12 and y < (pp_disp.shape[1] - 12):
                if pp_disp[y, x] > 25:
                    mode_window = pp_disp[y - 12:y + 13, x - 12:x + 13]
                    if mode_window.size > 0:
                        mode = stats.mode(mode_window.flatten())
                        pp_disp[y, x] = mode[0][0]

            # THRESHOLD
            if pp_disp[y, x] > 10:
                pp_disp[y, x] = 15

    print(f"Post-processing for disparity matrix of shape {disp_matrix.shape} complete.")

    return pp_disp

def create_point_cloud(image_path_left, image_path_right, height, width, window_size, search_range, output_txt_path):
    array_left = pre_process(image_path_left, height, width)
    array_right = pre_process(image_path_right, height, width)

    disp_matrix = calculate_disparity(array_left, array_right, window_size, search_range)
    proc_disp_matrix = post_process(disp_matrix)

    height, width = proc_disp_matrix.shape
    img_left = Image.open(image_path_left).resize((width, height))
    arr_left = np.array(img_left)

    xyzrgb = []

    for x in range(width):
        for y in range(height):
            z = np.multiply(proc_disp_matrix[y, x], 6)
            rgb = arr_left[y, x]
            xyzrgb.append([x, y, z, rgb[0], rgb[1], rgb[2]])

    df = pd.DataFrame(xyzrgb, columns=['x', 'y', 'z', 'r', 'g', 'b'])
    df.to_csv(output_txt_path, index=False)

    print(f"Successfully created file at {output_txt_path}.")

    return df

def create_ply(output_txt_path, output_ply_path):
    df = pd.read_csv(output_txt_path)
    with open(output_ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(df)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for _, row in df.iterrows():
            f.write(f"{row['x']} {row['y']} {row['z']} {row['r']} {row['g']} {row['b']}\n")

    print(f"Successfully created PLY file at {output_ply_path}.")

    return output_ply_path

def save_to_mlp(output_ply_path, output_mlp_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(output_ply_path)
    ms.save_project(output_mlp_path)
    print(f"MeshLab project saved to {output_mlp_path}.")

def main():
    image_path_left = 'koon_calibration/left/img10.png'
    image_path_right = 'koon_calibration/right/img10.png'
    height = 256
    width = 256
    window_size = 1
    search_range = 90
    output_txt_path = f"output/point_cloud_{height}.txt"
    output_ply_path = f"output/point_cloud_{height}.ply"
    output_mlp_path = f"output/point_cloud_{height}.mlp"

    create_point_cloud(image_path_left, image_path_right, height, width, window_size, search_range, output_txt_path)
    create_ply(output_txt_path, output_ply_path)
    save_to_mlp(output_ply_path, output_mlp_path)

    print(f"To open the result in MeshLab, open MeshLab and import the PLY file: {output_ply_path}")

if __name__ == "__main__":
    main()

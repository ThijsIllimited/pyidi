import glob
import sys
import os
current_dir = os.path.dirname(os.path.realpath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the current directory and parent directory to the system path
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Add the scripts directory to the system path
scripts_dir = os.path.join(current_dir, 'scripts')
sys.path.insert(0, scripts_dir)

import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EMA_functions import *

root_drive_sim = os.path.normpath('G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations')

# Find all files ending with .cih in the folder
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)
file_path_settings = 'I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv'

df_file_description = pd.read_csv(file_path_settings)

# if 'spider_ij_d' not in df_file_description.columns:
    # df_file_description['spider_ij_d'] = None
# if 'prey_ij_d' not in df_file_description.columns:
    # df_file_description['prey_ij_d'] = None

df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

fig, ax = plt.subplots(figsize=(15, 15))
plt.ion()
for file in files:
    file_name = file.split('\\')[-1]
    file_name_base = file_name.split('.')[0]
    file_root = file.split(file_name)[0]

    try:
        df_filtered = df_file_description[df_file_description['filename'].isin([file_name])]
        # spider_ij_d = df_filtered['spider_ij_d'].item()
        prey_ij_d = df_filtered['prey_ij_d'].item()
    except:
        continue
    
    # if not pd.isna(spider_ij_d):
    if not pd.isna(prey_ij_d):
        continue
    
    file_name_dist = os.path.join(file_root, f"{file_name_base}_dist.pkl")

    with open(file_name_dist, 'rb') as f:
        data = pkl.load(f)

    homography_matrix = data['homography_matrix']

    EMA_structure = EMA_Structure(file_name)
    video = EMA_structure.open_video(add_extension=False)
    still_frame = video.reader.get_frame(0)
    undistorted_frame = cv2.warpPerspective(still_frame, homography_matrix, (still_frame.shape[1], still_frame.shape[0]))
    implot = ax.imshow(undistorted_frame, cmap='gray')

    plt.show()

    # Capture user input by clicking a point in the figure
    print("Click a point in the figure")
    points = plt.ginput(1)  # Capture one point
    print(f"Selected point: {points}")
    
    # df_file_description.loc[df_file_description['filename'] == file_name, 'spider_ij_d'] = str(points[0])
    df_file_description.loc[df_file_description['filename'] == file_name, 'prey_ij_d'] = str(points[0])

    df_file_description.to_csv(file_path_settings)  
    implot.remove()


    
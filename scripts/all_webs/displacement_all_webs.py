import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
import numpy as np              # Python's standard numerical library
import matplotlib.pyplot as plt # Python's scientific visualization library
import pyidi                    # Python HSC data analysis library
import pickle as pk
from pixel_setter2 import PixelSetter#, play_video, detect_peaks
from pixel_setter import play_video
from scipy.ndimage import uniform_filter
import importlib
from EMA_functions import *
from Feature_selecter import *
import matplotlib.animation as animation
from pyidi import ROISelect
from matplotlib.path import Path
import time
import glob
import itertools
root_drive_sim = os.path.normpath('G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations')

# Find all files ending with .cih in the folder
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)

reference_image = (0, 150)
roi_size = (9,9)
bit_depth = 16
n_tracking_points = 3000

line_styles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a cycle iterator for line styles and colors
line_style_cycle = itertools.cycle(line_styles)
color_cycle = itertools.cycle(colors)

dic_type = '2D'
processors = 12 
for file in files:
    print(file)

    file_name = file.split('\\')[-1]
    file_name_base = file_name.split('.')[0]
    file_root = file.split(file_name)[0]
    save_file = os.path.join(file_root, f"{file_name_base}_analyzed.txt")

    if os.path.exists(save_file):
        continue
    
    file_name_points = os.path.join(file_root, f"{file_name_base}_points.npz")
    data = np.load(file_name_points)
    reference_image = data['referece_image']
    roi_size = data['roi_size']
    dic_type = data['type']
    points = data['points']
    n_tracking_points = points.shape[0]
    file_name_roi = os.path.join(file_root, f"{file_name_base}_ROI.npy")
    ROI = np.load(file_name_roi)
    path = Path(ROI.T)
    

    EMA_structure = EMA_Structure(file_name)
    video = EMA_structure.open_video(add_extension=False)
    mean_image = video.mraw[reference_image[0]:reference_image[1]].mean(axis=0)
    if dic_type == '2D':
        video.set_method('lk')
    elif dic_type == '1D':
        video.set_method('lk_1D')
    else:
        raise ValueError('Unknown tracking type')
    video.method.configure(roi_size = roi_size, reference_image = mean_image,  resume_analysis=False)
    video.set_points(points)
    video.get_displacements(processes = processors)

    with open(os.path.join(file_root, f"{file_name_base}_analyzed.txt"), 'w') as file:
        file.write(f"File: {file_name}\n")
        file.write(f"Reference Image: {reference_image}\n")
        file.write(f"ROI Size: {roi_size}\n")
        file.write(f"Tracking Type: {dic_type}\n")
        file.write(f"Number of Tracking Points: {n_tracking_points}\n")
        

    # fig, ax = plt.subplots(figsize=(15, 20))
    # ax.imshow(mean_image, cmap='gray')
    # ax.set_title(file_name_base + ' ' + str(n_tracking_points) + ' points')
    # ax.plot(ROI[1], ROI[0], label = 'ROI', linestyle='-', color='r')
    # points_plot = ax.scatter(points[:,1], points[:,0], marker='o',s=10)
    # ax.legend()
    # plt.show()


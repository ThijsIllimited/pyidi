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

for file in files:
    print(file)

    file_name = file.split('\\')[-1]
    file_name_base = file_name.split('.')[0]
    file_root = file.split(file_name)[0]

    file_name_points = os.path.join(file_root, f"{file_name_base}_points.npy")
    if os.path.exists(file_name_points):
        continue
    set_n_points = True
    file_name_roi = os.path.join(file_root, f"{file_name_base}_ROI.npy")
    ROI = np.load(file_name_roi)
    path = Path(ROI.T)
    
    # files_points = glob.glob('D:/thijsmas/HSC/**/*_points.npy', recursive=True)
    # files_name_base_points = ['_'.join(file_point.split('\\')[-1].split('_')[:-1]) for file_point in files_points]
    # len_files_roi = len(files_points)
    EMA_structure = EMA_Structure(file_name)
    video = EMA_structure.open_video(add_extension=False)
    mean_image = video.mraw[reference_image[0]:reference_image[1]].mean(axis=0)
    feature_selecter = FeatureSelecter(mean_image)
    feature_selecter.set_filter_method('eig0', roi_size)
    score_full = feature_selecter.apply_filter(downsample=1)

    mask_image = path.contains_points(np.array([(i,j) for i in range(mean_image.shape[0]) for j in range(mean_image.shape[1])])).reshape(mean_image.shape)
    mask_image = mask_image.reshape(mean_image.shape)

    score_full[~mask_image] = 0
    score_full[mean_image >= int(0.99*(2**bit_depth-1))] = 0

    points = feature_selecter.pick_max_loop(score_image = None, min_distance = (5,5), n_points = n_tracking_points, minimum_score= 10)
    score_list = score_full[points[:, 0], points[:, 1]]

    fig, ax = plt.subplots(figsize=(15, 20))
    ax.imshow(mean_image, cmap='gray')
    ax.set_title(file_name_base + ' ' + str(n_tracking_points) + ' points')
    ax.plot(ROI[1], ROI[0], label = 'ROI', linestyle='-', color='r')
    points_plot = ax.scatter(points[:,1], points[:,0], marker='o',s=10, c=score_list, cmap='RdYlGn')
    ax.legend()
    plt.ion()
    plt.show()
    
    while set_n_points:
        user_input = input("Please enter n_tracking_points: ")
        try:
            user_input = int(user_input)
            n_tracking_points = user_input
            points = feature_selecter.pick_max_loop(score_image = None, min_distance = (5,5), n_points = n_tracking_points, minimum_score= 10)
            score_list = score_full[points[:, 0], points[:, 1]]
            points_plot.remove()
            points_plot = ax.scatter(points[:,1], points[:,0], marker='o', s=10, c=score_list, cmap='RdYlGn')
            ax.set_title(file_name_base + ' ' + str(n_tracking_points) + ' points')
            plt.draw()
            plt.pause(0.001)
        except:
            set_n_points = False
    plt.close()
    data = {
        'points': points,
        'type': dic_type,
        'roi_size': roi_size,
        'referece_image': reference_image
    }
    np.savez(file_name_points, **data)

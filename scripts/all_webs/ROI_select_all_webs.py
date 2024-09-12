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

line_styles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a cycle iterator for line styles and colors
line_style_cycle = itertools.cycle(line_styles)
color_cycle = itertools.cycle(colors)

for file in files:
    print(file)

    file_name = file.split('\\')[-1]
    file_name_base = file_name.split('.')[0]
    file_root = file.split(file_name)[0]

    file_name_roi = os.path.join(file_root, f"{file_name_base}_ROI.npy")
    if os.path.exists(file_name_roi):
        continue
    
    files_roi = glob.glob('D:/thijsmas/HSC/**/*_ROI.npy', recursive=True)
    files_name_base_roi = ['_'.join(file_roi.split('\\')[-1].split('_')[:-1]) for file_roi in files_roi]
    len_files_roi = len(files_roi)
    EMA_structure = EMA_Structure(file_name)
    video = EMA_structure.open_video(add_extension=False)
    fig, ax = plt.subplots(figsize=(15, 20))
    for file_nr, (file_roi, file_name_base_roi) in enumerate(zip(files_roi, files_name_base_roi)):
        if file_nr < len_files_roi - 10:
            continue
        roi = np.load(file_roi)
        ax.plot(roi[1], roi[0], label = file_nr, linestyle=next(line_style_cycle), color=next(color_cycle))
    still_image = video.mraw[0]
    ax.imshow(still_image, cmap='gray')
    ax.legend()
    plt.show()

    user_input = input("Please enter the file base: ")
    try:
        user_input = int(user_input)
        polygon = np.load(os.path.join(files_roi[user_input]))
    except:
        roi_select = ROISelect(video)
        polygon = roi_select.polygon

    np.save(file_name_roi, polygon)
    print(os.path.join(file_root, f"{file_name_base}_ROI.npy"))
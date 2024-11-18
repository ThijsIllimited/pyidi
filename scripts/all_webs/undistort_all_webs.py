import os
import sys

current_dir = os.path.dirname(os.path.realpath('__file__'))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the current directory and parent directory to the system path
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Add the scripts directory to the system path
scripts_dir = os.path.join(current_dir, 'scripts')
sys.path.insert(0, scripts_dir)

from printing_tensioned_structures.src.network import *

import numpy as np              # Python's standard numerical library
import matplotlib.pyplot as plt # Python's scientific visualization library
import pyidi                    # Python HSC data analysis library
import pickle as pk
# from pixel_setter2 import PixelSetter#, play_video, detect_peaks
# from pixel_setter import play_video
# from scipy.ndimage import uniform_filter
# import importlib
from EMA_functions import *
from Feature_selecter import *
# import matplotlib.animation as animation
from pyidi import ROISelect
from matplotlib.path import Path
# import time
import glob
import itertools
import json
import pandas as pd
import cv2
root_drive_sim = os.path.normpath('G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations')

# Find all files ending with .cih in the folder
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)

df_file_description = pd.read_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

line_styles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a cycle iterator for line styles and colors
line_style_cycle = itertools.cycle(line_styles)
color_cycle = itertools.cycle(colors)

def import_net(file_name, json_root = r'D:\thijsmas'):
    file_path = os.path.join(json_root, file_name + '.json')
    with open(file_path) as f:
        data = json.load(f)
    vertices = []
    int_keys = [int(key) for key in data['vertices'].keys()]
    for key in sorted(int_keys):
        vertices.append([data['vertices'][str(key)]['x'], data['vertices'][str(key)]['y'], data['vertices'][str(key)]['z']])
    vertices = np.array(vertices)

    edges = []
    edge_keys = [int(key) for key in data['edges'].keys()]
    for key in sorted(edge_keys):
        edges.append(data['edges'][str(key)])

    # q = []
    # for key in sorted(data['q'].keys()):
        # q.append(data['q'][key])
    # q = np.array(q)

    fixed = data['fixed']

    # net = Network_custom.from_fd(vertices, edges, q, fixed, paths = None, dir = None)
    net = Network_custom()
    net.vertices = vertices
    net.edges = np.array(edges)
    # net.q = q
    net.fixed = data['fixed']
    net.center_node = data['gkey_center_n']["'node_number'"]
    return net

def undistorted_points_from_net(net, scaler, shift):
    return np.array([scaler*net.vertices[net.fixed + [net.center_node]][:, 0] + shift[0], -scaler*net.vertices[net.fixed + [net.center_node]][:, 1] - shift[1]], dtype=np.float32).T

scaler = 2.5

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
plt.ion()

for file in files:
    # Clear the axes
    ax[0].cla()
    ax[1].cla()
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[1].invert_yaxis()

    print(file)
    file_name = file.split('\\')[-1]
    file_name_base = file_name.split('.')[0]
    file_root = file.split(file_name)[0]

    file_name_dist = os.path.join(file_root, f"{file_name_base}_dist.pkl")
    if os.path.exists(file_name_dist):
        continue

    df_filtered = df_file_description[df_file_description['filename'].isin([file_name])]
    try:
        ecc = df_filtered['ecc'].item()
    except:
        print(f'No ecc found for {file_name}')
        continue
    file_name_mesh = f'Web_constant_length_ecc{ecc}_gap0_constraint'
    net = import_net(file_name_mesh)
    
    files_dist = glob.glob('D:/thijsmas/HSC/**/*_dist.pkl', recursive=True)
    files_name_base_dist = ['_'.join(file_dist.split('\\')[-1].split('_')[:-1]) for file_dist in files_dist]
    len_files_roi = len(files_dist)
    EMA_structure = EMA_Structure(file_name)
    video = EMA_structure.open_video(add_extension=False)
    
    still_image = video.reader.get_frame(0)
    height, width = still_image.shape[:2]
    shift = [width/2, -height/2]
    ax[0].imshow(still_image, cmap='gray')
    points_undistorted = undistorted_points_from_net(net, scaler, shift = shift)
    ax[1].plot(points_undistorted[:, 0], points_undistorted[:, 1],'r*', label = 'undistorted', zorder=10)
    for i in range(len(net.fixed)):
        ax[1].text(points_undistorted[i, 0], points_undistorted[i, 1], str(i), color='r', zorder=10)

    for file_nr, (file_dist, file_name_base_dist) in enumerate(zip(files_dist, files_name_base_dist)):
        if file_nr < len_files_roi - 10:
            continue
        with open(file_dist, 'rb') as f:
            data = pkl.load(f)
        points_distorted = data['points_distorted']
        ax[0].plot(points_distorted[:,0], points_distorted[:,1], label = file_nr, linestyle=next(line_style_cycle), color=next(color_cycle))

    ax[0].legend()
    
    plt.show()

    while True:
        user_input = input("Please enter the file base: ")
        try:
            user_input = int(user_input)
            with open(files_dist[user_input], 'rb') as f:
                data = pkl.load(f)
            points_distorted = data['points_distorted']
        except:
            roi_select = ROISelect(video)
            polygon = roi_select.polygon
            points_distorted = np.flip(np.array(polygon, dtype=np.float32).T[:-1], axis=1)
        
        try:
            homography_matrix, _ = cv2.findHomography(points_distorted, points_undistorted, method=cv2.RANSAC)
        except:
            continue
        undistorted_image = cv2.warpPerspective(still_image, homography_matrix, (width, height))
        undist_im = ax[1].imshow(undistorted_image, cmap='gray')
        user_input2 = input("Is the undistorted image correct? (y/n): ")
        if user_input2 == 'y':
            break
        else:
            undist_im.remove()
            plt.draw()

    data = {'homography_matrix': homography_matrix,
            'points_distorted': points_distorted,
            'scaler': scaler,
            'shift': shift, 
            'file_name_mesh': file_name_mesh,
            'file_name_base': file_name_base,
            'file_name': file_name,
            'ecc': ecc,
            }
    with open(file_name_dist, 'wb') as f:
        pk.dump(data, f)


    print(os.path.join(file_root, f"{file_name_base}_dist.npy"))
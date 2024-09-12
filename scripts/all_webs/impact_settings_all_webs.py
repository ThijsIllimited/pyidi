# imports
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
from pixel_setter import play_video
from scipy.ndimage import uniform_filter
import importlib
from EMA_functions import *
import matplotlib.animation as animation
from DIC_functions import *
import glob
import ast
# import re


# Import test data
df_file_description = pd.read_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)
# pattern = r'_S\d+\.cihx$'

# loop webs to set peak_n
invalid_files = []
for file_i, file in enumerate(files):
    # Get the file information
    name_video = os.path.basename(file)
    # if re.search(pattern, name_video):
    #     name_video_base = re.sub(pattern, '', name_video)
    root_video = os.path.dirname(file)
    df_filtered = df_file_description[df_file_description['filename'].isin([name_video])]
    indices = df_filtered.index
    try:
        peak_n = df_filtered['peak_n'].item()
        peak_F = df_filtered['peak_F'].item()
        peak_F_threshold = df_filtered['peak_F_threshold'].item()
    except:
        print(f'File {name_video} not found in file description')
        invalid_files.append(name_video)
        continue
    # nut_idx = df_filtered['nut_idx'].item()
    # d_lim = df_filtered['d_lim'].item()
    if isinstance(peak_n, str):
        if peak_n == 'taut' or peak_n == 'invalid test':
            continue
        peak_n = eval(peak_n)
        peak_F = eval(peak_F)
        peak_F_threshold = eval(peak_F_threshold)
    if not np.isnan(peak_n) and not np.isnan(peak_F) and not np.isnan(peak_F_threshold):
        if isinstance(peak_n, float):
            peak_n = int(peak_n)
            peak_F = float(peak_F)
            peak_F_threshold = float(peak_F_threshold)
            df_file_description.loc[indices, 'peak_n'] = peak_n
            df_file_description.loc[indices, 'peak_F'] = peak_F
            df_file_description.loc[indices, 'peak_F_threshold'] = peak_F_threshold
            df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
        continue
    
    if 'taut' in name_video:
        df_file_description.loc[indices, 'peak_n'] = 'taut'
        df_file_description.loc[indices, 'peak_F'] = 'taut'
        df_file_description.loc[indices, 'peak_F_threshold'] = 'taut'
        print(f'File {file_i} of {len(files)} saved')
        df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv') 
        continue

    EMA_structure = EMA_Structure(name_video)
    EMA_structure.open_impact_data()
    try:
        comment = EMA_structure.impact_data['comment']
        print(f"Comment: {comment}")
    except:
        print(f'File {name_video} not found in impact data')
        invalid_files.append(name_video)
        continue
    
    if comment == '':
        peak_n = 1
    elif 'first' in comment or 'First' in comment:
        peak_n = 1
    elif 'second' in comment or 'Second' in comment:
        peak_n = 2
    elif 'third' in comment or 'Third' in comment:
        peak_n = 3
    else:
        peak_n = 1
    
    peak_F_threshold = 0.5
    # Initialize signals
    EMA_structure.initialize_signals()
    try:
        first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)
    except:
        peak_n = 1
        peak_F_threshold = 0.05
        first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)
    peak_F = EMA_structure.F_peak

    fig, ax = plt.subplots(2,1, figsize = (10, 10))
    plt.ion()
    ax[0].set_title(f'Current peak_n: {peak_n},\n comment: {comment},\n peak_force: {peak_F},\n peak_F_threshold: {peak_F_threshold}')
    ax[0].plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot = ax[0].plot(EMA_structure.t_force_raw[first_zero_id_force], EMA_structure.force_raw[first_zero_id_force], 'r*')
    ax[1].plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot_zoom = ax[1].plot(EMA_structure.t_force_raw[first_zero_id_force], EMA_structure.force_raw[first_zero_id_force], 'r*')
    ax[1].set_xlim(EMA_structure.t_force_raw[first_zero_id_force]-0.005, EMA_structure.t_force_raw[first_zero_id_force]+0.005)
    plt.show()

    # User input
    while True:
        input_string = input("'c' to continue\n 't' to reset approximate peak height, \n New peak_n:...,  \n")
        if input_string == 'c':
            break
        elif input_string == 't':
            input_string = input("New peak threshold:...,  \n")
            peak_F_threshold = float(input_string)
        elif input_string == 'q':
            peak_n = 'invalid test'
            peak_F = 'invalid test'
            peak_F_threshold = 'invalid test'
            break
        else:
            peak_n = int(input_string)
        first_zero_id_force_new = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold = 0.05, approximate_height = peak_F_threshold)
        peak_F = EMA_structure.F_peak
        ax[0].set_title(f'Current peak_n: {peak_n},\n comment: {comment},\n peak_force: {peak_F},\n peak_F_threshold: {peak_F_threshold}')
        start_plot[0].set_data(EMA_structure.t_force_raw[first_zero_id_force_new], EMA_structure.force_raw[first_zero_id_force_new])
        start_plot_zoom[0].set_data(EMA_structure.t_force_raw[first_zero_id_force_new], EMA_structure.force_raw[first_zero_id_force_new])
        ax[1].set_xlim(EMA_structure.t_force_raw[first_zero_id_force_new]-0.005, EMA_structure.t_force_raw[first_zero_id_force_new]+0.005)
        plt.draw()
        plt.pause(0.001)
    plt.close(fig)
    df_file_description.loc[indices, 'peak_n'] = peak_n
    df_file_description.loc[indices, 'peak_F'] = peak_F
    df_file_description.loc[indices, 'peak_F_threshold'] = peak_F_threshold
    
    print(f'File {file_i} of {len(files)} saved')
    df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')

print(invalid_files)
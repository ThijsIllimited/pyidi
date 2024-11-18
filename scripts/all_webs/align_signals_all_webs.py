# imports
import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the current directory and parent directory to the system path
sys.path.insert(0, current_dir)
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


# Import test data
df_file_description = pd.read_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

if 'id0_cam' not in df_file_description.columns:
    df_file_description['id0_cam'] = None
if 'id0_for' not in df_file_description.columns:
    df_file_description['id0_for'] = None

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)

# loop webs to set peak_n
invalid_files = []

fig, ax = plt.subplots(4, 1, figsize=(15, 10))
plt.ion()
for file_i, file in enumerate([files[0]]):
    # Get the file information
    name_video = os.path.basename(file)
    root_video = os.path.dirname(file)
    df_filtered = df_file_description[df_file_description['filename'].isin([name_video])]
    indices = df_filtered.index

    try:
        peak_n = ast.literal_eval(df_filtered['peak_n'].item())
        peak_F = ast.literal_eval(df_filtered['peak_F'].item())
        peak_F_threshold = ast.literal_eval(df_filtered['peak_F_threshold'].item())
        nut_idx = df_filtered['nut_idx'].item()
        spider_ij = ast.literal_eval(df_filtered['spider_ij'].item())
        prey_ij = ast.literal_eval(df_filtered['prey_ij'].item())
        shift = ast.literal_eval(df_filtered['shift'].item())
        d_lim = df_filtered['d_lim'].item()
        test_number = int(df_filtered['test_number'].item())
    except:
        print(f'Something went wrong with {name_video}')
        continue
    id0_cam = df_filtered['id0_cam'].item()
    if not pd.isna(id0_cam):
        print('Already done')
        continue

    EMA_structure = EMA_Structure(name_video)
    DIC_structure = DIC_Structure(file)
    df = DIC_structure.list_test_data(test_range = range(1, 100), robostness_check = False)
    EMA_structure.tp, EMA_structure.d = DIC_structure.join_results([test_number])
    td = EMA_structure.d +  EMA_structure.tp.reshape(len(EMA_structure.tp),1,2)

    EMA_structure.open_impact_data()
    video = EMA_structure.open_video(add_extension = False)
    EMA_structure.initialize_signals()
    EMA_structure.initialize_displacement(idx='all', dir='xy')
    EMA_structure.nut_idx((prey_ij[0] + shift[0], prey_ij[1] + shift[1]), exclude_high_amplitude = True, d_lim = d_lim)
    comment = EMA_structure.impact_data['comment']
    
    max_d = np.max(np.abs(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0]))
    first_zero_id_cam = EMA_structure.find_signal_start(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0], treshold=0.08, approximate_height = max_d*.5, approximate_distance=100000)
    first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)

    t_camera = EMA_structure.t_camera_raw - EMA_structure.t_camera_raw[first_zero_id_cam]
    t_force = EMA_structure.t_force_raw - EMA_structure.t_force_raw[first_zero_id_force]

    ax[0].set_title(f'Current peak_n: {peak_n},\n comment: {comment},\n peak_force: {peak_F},\n peak_F_threshold: {peak_F_threshold}')
    ax[0].plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot = ax[0].plot(EMA_structure.t_force_raw[first_zero_id_force], EMA_structure.force_raw[first_zero_id_force], 'r*')
    ax[1].plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot_zoom = ax[1].plot(EMA_structure.t_force_raw[first_zero_id_force], EMA_structure.force_raw[first_zero_id_force], 'r*')
    ax[1].set_xlim(EMA_structure.t_force_raw[first_zero_id_force]-0.005, EMA_structure.t_force_raw[first_zero_id_force]+0.005)
    ax[1].set_title(name_video)


    ax[2].set_title(f'shift: {shift}, id force: {first_zero_id_force}, id cam: {first_zero_id_cam}')
    ax[2].plot(t_camera, EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0])
    ax[2].plot(t_camera[first_zero_id_cam], EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,first_zero_id_cam,0], 'r*')


    plt.draw()

    # User input
    set_displacement_id = False
    while True:
        input_string = input("'c' to continue\n (.. ,..)  = (peak_n, approximate peak height), \n 'd' continue and shift displacements  \n")
        if input_string == 'c':
            break
        elif input_string == 'd':
            set_displacement_id = True
            break
        else:
            try:
                peak_n, peak_F_threshold = ast.literal_eval(input_string)
            except:
                print('Invalid input')
                continue
        first_zero_id_force_new = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold = 0.05, approximate_height = peak_F_threshold)
        peak_F = EMA_structure.F_peak
        ax[0].set_title(f'Current peak_n: {peak_n},\n comment: {comment},\n peak_force: {peak_F},\n peak_F_threshold: {peak_F_threshold}')
        start_plot[0].set_data([EMA_structure.t_force_raw[first_zero_id_force_new]], [EMA_structure.force_raw[first_zero_id_force_new]])
        start_plot_zoom[0].set_data([EMA_structure.t_force_raw[first_zero_id_force_new]], [EMA_structure.force_raw[first_zero_id_force_new]])
        ax[1].set_xlim(EMA_structure.t_force_raw[first_zero_id_force_new]-0.005, EMA_structure.t_force_raw[first_zero_id_force_new]+0.005)
        plt.draw()

    while set_displacement_id:


    df_file_description.loc[indices, 'peak_n'] = peak_n
    df_file_description.loc[indices, 'peak_F'] = peak_F
    df_file_description.loc[indices, 'peak_F_threshold'] = peak_F_threshold
    print(f'File {file_i} of {len(files)} saved')

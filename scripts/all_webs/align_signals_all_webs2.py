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
import matplotlib.gridspec as gridspec
# import re

# Import test data
file_path_settings = 'I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv'
df_file_description = pd.read_csv(file_path_settings)
df_file_description = df_file_description.loc[:, ~df_file_description.columns.str.startswith('Unnamed')]
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# Import test data
df_file_description = pd.read_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# if 'id0_cam' not in df_file_description.columns:
#     df_file_description['id0_cam'] = None
# if 'id0_for' not in df_file_description.columns:
#     df_file_description['id0_for'] = None

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
    # if name_video == 'Full_web_ecc1_new0_rev2_Floc17_v0_S01.cihx':
    #     df_file_description.loc[16, 'nut_idx'] = None
    #     continue
    try:
        peak_n = df_filtered['peak_n'].item()
        peak_F = df_filtered['peak_F'].item()
        peak_F_threshold = df_filtered['peak_F_threshold'].item()
        prey_ij = ast.literal_eval(df_filtered['prey_ij'].item())
        spider_ij = ast.literal_eval(df_filtered['spider_ij'].item())
    except:
        print(f'File {name_video} not found in file description')
        invalid_files.append(name_video)
        continue
    if isinstance(peak_n, str):
        if peak_n == 'taut' or peak_n == 'invalid test':
            print(f'File {name_video} is a taut or invalid test')
            continue
        peak_n = eval(peak_n)
        peak_F = eval(peak_F)
        peak_F_threshold = eval(peak_F_threshold)
    
    try:
        shift = ast.literal_eval(df_filtered['shift'].item())
        d_lim = df_filtered['d_lim'].item()
        test_number = int(df_filtered['test_number'].item())
        nut_idx = int(df_filtered['nut_idx'].item())
    except:
        print(f'd_lim not set in {name_video} (EMA_settings_all_webs.py)')

    if not pd.isna(df_filtered['id0_cam'].item()):
        print('Already done')
        continue

    EMA_structure = EMA_Structure(name_video)
    # Open impact data
    EMA_structure.open_impact_data()
    video = EMA_structure.open_video(add_extension = False)
    try:
        comment = EMA_structure.impact_data['comment']
        print(f"Comment: {comment}")
    except:
        print(f'File {name_video} not found in impact data')
        invalid_files.append(name_video)
        continue
    
    # Open displacement data
    DIC_structure = DIC_Structure(file)
    # video = DIC_structure.video
    df = DIC_structure.list_test_data(test_range = range(1, 100), robostness_check = False)
    print(df)
    try:
        last_row = df.iloc[-1]  # Get the last row of the DataFrame
    except:
        print(f'File {name_video} not found in DIC data')
        invalid_files.append(name_video)
        continue
    test_number = last_row['test_number']
    # input_string = input(f"'' to continue with test_number: {test_number}, type integer to change test_number: ")
    # if input_string != '':
    #     test_number = int(input_string)
    print(f"Test number: {test_number}")
    EMA_structure.tp, EMA_structure.d = DIC_structure.join_results([test_number])
    td = EMA_structure.d +  EMA_structure.tp.reshape(len(EMA_structure.tp),1,2)

    EMA_structure.initialize_signals()
    EMA_structure.initialize_displacement(idx='all', dir='xy')
    EMA_structure.nut_idx((prey_ij[0] + shift[0], prey_ij[1] + shift[1]), exclude_high_amplitude = True, d_lim = d_lim)

    # First find the zero crossing automatically. This can be updated later if the user wants to change it
    max_d = np.max(np.abs(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0]))
    id0_cam = EMA_structure.find_signal_start(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0], treshold=0.08, approximate_height = max_d*.5, approximate_distance=100000)
    id0_for = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)
    EMA_structure.process_signals(id0_cam, id0_for)

    fig1 = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1])

    # Add subplots to the first column (4 rows)
    ax1 = fig1.add_subplot(gs[0, 0]) # ax1 contains the displacement signal (y)
    ax2 = fig1.add_subplot(gs[1, 0]) # ax2 contains the displacement signal (y) zoomed in
    ax3 = fig1.add_subplot(gs[2, 0]) # ax3 contains the force signal
    ax4 = fig1.add_subplot(gs[3, 0]) # ax4 contains the force signal zoomed in
    ax5 = fig1.add_subplot(gs[:2, 1]) # ax5 contains the first frame with the nut and prey location
    ax6 = fig1.add_subplot(gs[2:, 1])  # ax6 contains the first frame with the valid points

    ax5.imshow(video.reader.get_frame(0), cmap='gray', vmin=0, vmax=2**16-1)
    ax5.set_xticks([])
    ax5.set_yticks([])
    nut_loc_plot = ax5.plot(EMA_structure.tp[EMA_structure.nearest_nut_index,1], EMA_structure.tp[EMA_structure.nearest_nut_index,0], 'r*')
    ax5.plot(spider_ij[0], spider_ij[1], 'ro')
    ax5.plot(prey_ij[0], prey_ij[1], 'bo')

    ax6.imshow(video.reader.get_frame(0), cmap='gray', vmin=0, vmax=2**16-1)
    ax6.set_xticks([])
    ax6.set_yticks([])
    valid_points_plot = ax6.plot(EMA_structure.tp[EMA_structure.exclude_high_amplitude,1], EMA_structure.tp[EMA_structure.exclude_high_amplitude,0], 'r.')
    ax1.set_title(f'{name_video} \n Current id0_for: {id0_for} \n Current id0_cam: {id0_cam} \n shift: {shift}')
    nut_plot_y = ax1.plot(EMA_structure.t_camera_raw, EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0], label=f'Node {EMA_structure.nearest_nut_index}')
    nut_plot_y_marker = ax1.plot(EMA_structure.t_camera_raw[id0_cam], EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,id0_cam,0], 'r*')
    nut_plot_yzoom = ax2.plot(EMA_structure.t_camera, EMA_structure.displacements_y[EMA_structure.nearest_nut_index,:], label=f'Node {EMA_structure.nearest_nut_index}')
    ax2.set_xlim(0, 0.1)
    ax3.plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot = ax3.plot(EMA_structure.t_force_raw[id0_for], EMA_structure.force_raw[id0_for], 'r*')
    ax4.plot(EMA_structure.t_force, EMA_structure.force)
    # start_plot_zoom = ax4.plot(EMA_structure.t_force_raw[id0_for], EMA_structure.force_raw[id0_for], 'r*')
    ax4.set_xlim(0, 0.01)
    
    plt.ion()
    plt.show()

    while True:
        input_string = input(f"Enter shifts (id0_for, id0_cam), 's' to shift location, or 'q' to quit: ")
        if input_string == 'q' or input_string == '':
            break
        if input_string == 's':
            input_string = input(f"Enter shift: '(i, j)' ")
            shift = ast.literal_eval(input_string)
            EMA_structure.nut_idx((prey_ij[0] + shift[0], prey_ij[1] + shift[1]), exclude_high_amplitude = True, d_lim = d_lim)
            nut_loc_plot[0].set_data([EMA_structure.tp[EMA_structure.nearest_nut_index,1]], [EMA_structure.tp[EMA_structure.nearest_nut_index,0]])
            continue
        try:
            id0_for_delta, id0_cam_delta = ast.literal_eval(input_string)
            id0_for += id0_for_delta
            id0_cam += id0_cam_delta
        except:
            print('Invalid input')
            continue
        EMA_structure.initialize_signals()
        EMA_structure.initialize_displacement(idx='all', dir='xy')
        EMA_structure.nut_idx((prey_ij[0] + shift[0], prey_ij[1] + shift[1]), exclude_high_amplitude = True, d_lim = d_lim)
        EMA_structure.process_signals(id0_cam, id0_for)
        ax1.set_title(f'{name_video} \n Current id0_for: {id0_for} \n Current id0_cam: {id0_cam} \n shift: {shift}')

        nut_loc_plot[0].set_data([EMA_structure.tp[EMA_structure.nearest_nut_index,1]], [EMA_structure.tp[EMA_structure.nearest_nut_index,0]])
        nut_plot_y[0].set_data(EMA_structure.t_camera_raw, EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0])
        nut_plot_y_marker[0].set_data([EMA_structure.t_camera_raw[id0_cam]], [EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,id0_cam,0]])
        nut_plot_yzoom[0].set_data(EMA_structure.t_camera, EMA_structure.displacements_y[EMA_structure.nearest_nut_index,:])
        start_plot[0].set_data([EMA_structure.t_force_raw[id0_for]], [EMA_structure.force_raw[id0_for]])
        valid_points_plot[0].set_data(EMA_structure.tp[EMA_structure.exclude_high_amplitude,1], EMA_structure.tp[EMA_structure.exclude_high_amplitude,0])
    plt.close('all')

    df_file_description.loc[indices, 'shift'] = str(shift)
    df_file_description.loc[indices, 'nut_idx'] = EMA_structure.nearest_nut_index
    df_file_description.loc[indices, 'id0_for'] = id0_for
    df_file_description.loc[indices, 'id0_cam'] = id0_cam
    
    print(f'File {file_i} of {len(files)-1} saved')
    df_file_description.to_csv(file_path_settings)


print(invalid_files)
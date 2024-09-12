# imports
import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
import numpy as np              # Python's standard numerical library
import matplotlib.pyplot as plt # Python's scientific visualization library
# from pixel_setter import play_video
# from scipy.ndimage import uniform_filter
import importlib
from EMA_functions import *
from DIC_functions import *
import glob
import ast
from sdypy import EMA
import pickle as pkl
# import re

# Import test data
df_file_description = pd.read_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)
# pattern = r'_S\d+\.cihx$'

lower = 5
higher = 300
pol_order_high = 150

# loop webs to set peak_n
invalid_files = []
for file_i, file in enumerate(files):
    # Get the file information
    name_video = os.path.basename(file)
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
        shift = ast.literal_eval(df_filtered['shift'].item())
        d_lim = df_filtered['d_lim'].item()
        test_number = df_filtered['test_number'].item()
        nut_idx = df_filtered['nut_idx'].item()
        smooth_lim = df_filtered['smooth_lim'].item()
        max_drift = df_filtered['max_drift'].item()
        max_end_drift = df_filtered['max_end_drift'].item()
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
    if np.isnan(peak_n):
        print(f'File {name_video} has no peak_n')
        continue
    test_number = int(test_number)
    nut_idx = int(nut_idx)

    
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
    
    file_path_cam = os.path.join(root_video, f'{name_video}_cam.pkl')
    file_path_camd = os.path.join(root_video, f'{name_video}_cam_damped.pkl')
    file_path_EMA = os.path.join(root_video, f'{name_video}_EMA.pkl')
    file_path_EMAd = os.path.join(root_video, f'{name_video}_EMA_damped.pkl')

    if os.path.exists(file_path_EMAd):
        print(f'File {name_video} already exists')
        # os.remove(file_path_camd)
        # os.remove(file_path_EMAd)
        continue

    EMA_structure.tp, EMA_structure.d = DIC_structure.join_results([test_number])
    td = EMA_structure.d +  EMA_structure.tp.reshape(len(EMA_structure.tp),1,2)
    
    EMA_structure.initialize_signals()
    EMA_structure.initialize_displacement(idx='all', dir='xy')
    EMA_structure.nut_idx((prey_ij[0] + shift[0], prey_ij[1] + shift[1]), exclude_high_amplitude = True, d_lim = d_lim)

    max_d = np.max(np.abs(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0]))
    first_zero_id_cam = EMA_structure.find_signal_start(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0], treshold=0.08, approximate_height = max_d*.5, approximate_distance=100000)
    first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)
    # first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05)
    zeros_camera    = EMA_structure.greatest_common_divisor(EMA_structure.fs_camera, EMA_structure.fs_force)
    zeros_force     = EMA_structure.n_samples_camera_to_force(zeros_camera)

    # Shift and align signals
    EMA_structure.t_camera = EMA_structure.shift_time(EMA_structure.t_camera_raw, EMA_structure.t_camera_raw[first_zero_id_cam-zeros_camera])
    EMA_structure.t_force = EMA_structure.shift_time(EMA_structure.t_force_raw, EMA_structure.t_force_raw[first_zero_id_force-zeros_force])

    # Clip signals
    EMA_structure.t_camera          = EMA_structure.clip_signal_before(EMA_structure.t_camera,first_zero_id_cam-zeros_camera)
    # EMA_structure.displacements     = EMA_structure.clip_signal_before(EMA_structure.displacements_raw, first_zero_id_cam-zeros_camera)
    EMA_structure.displacements_x     = EMA_structure.clip_signal_before(EMA_structure.displacements_raw[:,:,1], first_zero_id_cam-zeros_camera)
    EMA_structure.displacements_y     = EMA_structure.clip_signal_before(EMA_structure.displacements_raw[:,:,0], first_zero_id_cam-zeros_camera)
    EMA_structure.t_force           = EMA_structure.clip_signal_before(EMA_structure.t_force, first_zero_id_force-zeros_force)
    EMA_structure.force             = EMA_structure.clip_signal_before(EMA_structure.force_raw, first_zero_id_force-zeros_force)

    last_id_force, last_id_camera   = EMA_structure.find_last_common_time_ids(EMA_structure.t_camera, EMA_structure.t_force)
    EMA_structure.t_camera          = EMA_structure.clip_signal_after(EMA_structure.t_camera, last_id_camera)
    # EMA_structure.displacements     = EMA_structure.clip_signal_after(EMA_structure.displacements, last_id_camera)
    EMA_structure.displacements_x     = EMA_structure.clip_signal_after(EMA_structure.displacements_x, last_id_camera)
    EMA_structure.displacements_y     = EMA_structure.clip_signal_after(EMA_structure.displacements_y, last_id_camera)
    EMA_structure.t_force           = EMA_structure.clip_signal_after(EMA_structure.t_force, last_id_force)
    EMA_structure.force             = EMA_structure.clip_signal_after(EMA_structure.force, last_id_force)

    # Zero Force after impact
    EMA_structure.force = EMA_structure.zero_signal_treshold(EMA_structure.force, 0.1)

    EMA_structure.set_freq_properties(padding_ratio=1)
    EMA_structure.get_transfer_function(direction='y')

    smooth_signals = np.max(np.abs(np.diff(np.linalg.norm(EMA_structure.d, axis=2))), axis = 1) < smooth_lim
    non_drifting = np.abs(np.mean(np.linalg.norm(EMA_structure.d[:,:-100], axis=2), axis=1) < max_drift)
    non_drifting2 = np.abs(np.linalg.norm(EMA_structure.d[:,-1], axis=1)) < max_end_drift
    EMA_structure.valid_tps = smooth_signals & non_drifting & non_drifting2 & EMA_structure.exclude_high_amplitude

    cam = EMA.Model(EMA_structure.H1[EMA_structure.valid_tps], EMA_structure.freq_camera, lower=5, upper=300, pol_order_high=150, frf_type = 'receptance')
    cam.get_poles(show_progress=True)
    # with open(file_path_cam, 'wb') as f:
    #     pkl.dump(cam, f)
    # EMA_structure.damping_ratio = 0
    # with open(file_path_EMA, 'wb') as f:
    #     pkl.dump(EMA_structure, f)

    EMA_structure.damping_ratio = 0.15
    time_windows = np.linspace(0, 1, len(EMA_structure.t_camera))
    exponential_w = np.exp(np.log(EMA_structure.damping_ratio) * time_windows)
    EMA_structure.set_freq_properties(padding_ratio=3)
    EMA_structure.displacements_x = EMA_structure.displacements_x*exponential_w
    EMA_structure.displacements_y = EMA_structure.displacements_y*exponential_w
    EMA_structure.get_transfer_function(direction='y')
    
    cam = EMA.Model(EMA_structure.H1[EMA_structure.valid_tps], EMA_structure.freq_camera, lower=5, upper=300, pol_order_high=150, frf_type = 'receptance')
    cam.get_poles(show_progress=True)
    with open(file_path_camd, 'wb') as f:
        pkl.dump(cam, f)
    with open(file_path_EMAd, 'wb') as f:
        pkl.dump(EMA_structure, f)
    
    df_file_description.loc[indices, 'lower'] = lower
    df_file_description.loc[indices, 'higher'] = higher
    df_file_description.loc[indices, 'pol_order_high'] = pol_order_high
    df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
print(invalid_files)
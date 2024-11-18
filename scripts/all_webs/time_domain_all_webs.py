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
# from pixel_setter import play_video
# from scipy.ndimage import uniform_filter
import importlib
from EMA_functions import *
from DIC_functions import *
import glob
import ast
from sdypy import EMA
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.gridspec as gridspec
# import re

# Import test data
df_file_description = pd.read_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
df_file_description = df_file_description.loc[:, ~df_file_description.columns.str.startswith('Unnamed')]
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)
# pattern = r'_S\d+\.cihx$'

lower = 5
higher = 300
pol_order_high = 150

D_in = 40
D_out = 100
nut_wh = (15, 60)
required_parameters = [ 'id0_cam',
                        'id0_for',
                        'double_tap',
                        'smooth_lim',
                        'max_drift',
                        'max_end_drift', 
                        'peak_n', 
                        'peak_F', 
                        'peak_F_threshold', 
                        'shift',
                        'test_number', 
                        'd_lim', 
                        'prey_ij_d', 
                        'prey_ij', 
                        'spider_ij_d', 
                        'spider_ij']
# loop webs to set peak_n
invalid_files = []
for file_i, file in enumerate(files):
    # Get the file information
    name_video = os.path.basename(file)
    root_video = os.path.dirname(file)
    
    file_parameters, df_index = unpack_dataframe(df_file_description, name_video, required_parameters)

    if file_parameters is None:
        print(f'some items in {name_video} could not be unpacked')
        invalid_files.append(name_video)
        continue

    EMA_structure = EMA_Structure(name_video)
    EMA_structure.set_params(**file_parameters)
    EMA_structure.set_params(FN0 = 41.2, reaction_time = 0.1) # FN0 from Lott: Prey localization in spider orb webs using modal vibration analysis. reaction_time from (Klärner and Barth1982)
    
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

    if os.path.exists(file_path_cam):
        cam = pkl.load(open(file_path_cam, 'rb'))
        normalized = True
        try:
            if cam.nat_freq[0]==0:
                EMA_structure.fn0 = cam.nat_freq[1]
            else:
                EMA_structure.fn0 = cam.nat_freq[0]
        except:
            normalized = False
            EMA_structure.fn0 = 0
    else:
        normalized = False
        EMA_structure.fn0 = 0

    if not normalized:
        print(f'File {name_video} cannot be normalized')
        continue

    EMA_structure.tp, EMA_structure.d = DIC_structure.join_results([EMA_structure.test_number])
    td = EMA_structure.d +  EMA_structure.tp.reshape(len(EMA_structure.tp),1,2)
    
    EMA_structure.initialize_signals()
    EMA_structure.initialize_displacement(idx='all', dir='xy')
    EMA_structure.nut_idx((EMA_structure.prey_ij[0] + EMA_structure.shift[0], EMA_structure.prey_ij[1] + EMA_structure.shift[1]), exclude_high_amplitude = True, d_lim = EMA_structure.d_lim)

    EMA_structure.process_signals(EMA_structure.id0_cam, EMA_structure.id0_for)

    EMA_structure.set_freq_properties(padding_ratio=1)
    EMA_structure.get_transfer_function(direction='y')

    smooth_signals = np.max(np.abs(np.diff(np.linalg.norm(EMA_structure.d, axis=2))), axis = 1) < EMA_structure.smooth_lim
    non_drifting = np.abs(np.mean(np.linalg.norm(EMA_structure.d[:,:-100], axis=2), axis=1) < EMA_structure.max_drift)
    non_drifting2 = np.abs(np.linalg.norm(EMA_structure.d[:,-1], axis=1)) < EMA_structure.max_end_drift
    EMA_structure.valid_tps = smooth_signals & non_drifting & non_drifting2 & EMA_structure.exclude_high_amplitude

    fig = plt.figure(figsize=(13, 18))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    gs = gridspec.GridSpec(4, 2, height_ratios=[.5, 1, 1, 1], width_ratios=[.7, 1])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left subplot
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right subplot
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[1, :])  # 
    ax4 = fig.add_subplot(gs[2, :])  # 
    ax5 = fig.add_subplot(gs[3, :])  # Bottom subplot spanning both columns
    # fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    fig, ax1 = EMA_structure.plot_hub_features(fig, ax1, DIC_structure.video, D_in = D_in, D_out = D_out, nut_wh = nut_wh, n_sections = 8)
    fig, ax3, t_max_vec, d_max_vec = EMA_structure.plot_hub_disp(fig, ax3, normalized = normalized, style = 'translation', t_max=0.8, legend = True, title = False, lim_lines = True, plot_nut_disp=True)
    max_t_diff = np.max(t_max_vec) - np.min(t_max_vec)
    max_d_diff = np.max(d_max_vec) - np.min(d_max_vec)
    fig, ax4, _, _ = EMA_structure.plot_hub_disp(fig, ax4, normalized = normalized, style = 'translation', t_max=0.15, legend = False, title = False)
    # fig, ax5, _, _ = EMA_structure.plot_hub_disp(fig, ax5, normalized = normalized, style = 'rotation', t_max=0.8, legend = True, title = False)
    fig, ax5, _, _ = EMA_structure.plot_hub_disp(fig, ax5, normalized = normalized, style = 'pitch_roll', t_max=0.8, legend = True, title = False)
    info_text0 = f"""
            Figure Information:
            - fn0: {EMA_structure.fn0:.2f} Hz
            - FN0: {EMA_structure.FN0:.2f} Hz
            - Reaction time: {EMA_structure.reaction_time:.2f} s
            - Peak Force: {EMA_structure.peak_F:.2f} N
            """
    info_text1 = f"""
            - Test number: {EMA_structure.test_number}
            - Prey distance: {np.linalg.norm([EMA_structure.prey_ij_dist[0]-EMA_structure.spider_ij_dist[0], EMA_structure.prey_ij_dist[1] - EMA_structure.spider_ij_dist[1]]):.2f} px
            - Normalized?: {normalized}
            - delta T: {max_t_diff:.2f} s
            - delta D: {max_d_diff:.2f} px/-
            """
    info_text2 = f"""
            - D_in: {D_in} px
            - D_out: {D_out} px
            - Nut width: {nut_wh[0]} px
            - Nut height: {nut_wh[1]} px
            - double tap: {EMA_structure.double_tap}
            """
    ax2.text(1/6, 1, info_text0, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')
    ax2.text(3/6, 1, info_text1, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')
    ax2.text(5/6, 1, info_text2, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')
    fig.suptitle(EMA_structure.file_name)
    # plt.show()
    print(f'Figure for {name_video} is generated')
    fig.savefig(f'{root_video}/{name_video}_hub_disp.png')

    df_file_description.loc[df_index, 'max_t_diff'] = max_t_diff
    df_file_description.loc[df_index, 'max_d_diff'] = max_d_diff
    df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')

print(invalid_files)
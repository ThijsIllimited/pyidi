# imports
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
import numpy as np              # Python's standard numerical library
import matplotlib.pyplot as plt # Python's scientific visualization library
# from pixel_setter import play_video
from EMA_functions import *
from DIC_functions import *
import glob
from sdypy import EMA
from sdypy.EMA import stabilization
import pickle as pkl
from scipy.signal import find_peaks


import matplotlib.pyplot as plt
import numpy as np

file_path_settings = 'I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv'
df_file_description = pd.read_csv(file_path_settings)
# Back up the data
df_file_description.to_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# List all files
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
                        'spider_ij',
                        'max_t_diff']


double_tap = False
taut_loose = 'Loose'
impact_pluck = 'Impact'
df_filtered = df_file_description[(df_file_description['taut/loose'] == taut_loose) &
                                   (df_file_description['impact/pluck'] == impact_pluck) &
                                   (df_file_description['double_tap'] == double_tap)]

for file_i, (name_video, path_video, root_video) in enumerate(zip(df_filtered['filename'], df_filtered['path2'], df_filtered['root'])):
    if pd.isna(path_video):
        print(f'Could not open {name_video}')
        continue
    df = df_file_description[df_file_description['filename'].isin([name_video])]
    try:
        peak_n = df['peak_n'].item()
        peak_F = df['peak_F'].item()
        peak_F_threshold = df['peak_F_threshold'].item()
        prey_ij = ast.literal_eval(df['prey_ij'].item())
        spider_ij = ast.literal_eval(df['spider_ij'].item())
        shift = ast.literal_eval(df['shift'].item())
        d_lim = df['d_lim'].item()
        test_number = df['test_number'].item()
        nut_idx = df['nut_idx'].item()
        smooth_lim = df['smooth_lim'].item()
        max_drift = df['max_drift'].item()
        max_end_drift = df['max_end_drift'].item()
    except:
        print(f'File {name_video} not found in file description')
        continue
    if os.path.exists(f'{root_video}/{name_video}_main_modes.png'):
        print(f'{name_video} already has a figure')
        continue
    file_parameters, index = unpack_dataframe(df_filtered, name_video, required_parameters)
    if file_parameters is None:
        print(f'some items in {name_video} could not be unpacked')
        continue
    EMA_structure = EMA_Structure(name_video)
    EMA_structure.set_params(**file_parameters)
    if np.isnan(EMA_structure.id0_cam):
        print(f'{name_video} has no cam id')
        continue

    EMA_structure.set_params(FN0 = 41.2, reaction_time = 0.1) # FN0 from Lott: Prey localization in spider orb webs using modal vibration analysis. reaction_time from (Klärner and Barth1982)
    if EMA_structure.double_tap:
        print(f'{name_video} had a double impact')
        continue

    EMA_structure.open_impact_data()
    video = EMA_structure.open_video(add_extension = False)
    fps = video.info['Record Rate(fps)']
    dt = 1/fps
                    
    DIC_structure = DIC_Structure(path_video)
    df = DIC_structure.list_test_data(test_range = range(1, 100), robostness_check = False)
    print(df)
    try:
        last_row = df.iloc[-1]  # Get the last row of the DataFrame
    except:
        print(f'File {name_video} not found in DIC data')
        continue

    file_path_cam = os.path.join(root_video, f'{name_video}_cam_2.pkl')
    with open(file_path_cam, 'rb') as file:
        cam = pkl.load(file)

    if 'nat_freq' in cam.__dict__.keys():
        continue

    try:
        file_name = file_path_cam.split('\\')[-1].split('.cihx_cam.pkl')[0]
    except:
        file_name = file_path_cam.split('\\')[-1].split('_cam.pkl')[0]
    # frf_d_raw_plot.set_data(cam_d.freq, np.mean(np.abs(cam_d.frf), axis=0))
    plt.draw()
    plt.pause(0.001)
    while True:
        # cam.select_poles(approx_nat_freq=EMA_structure_d.freq_camera[peaks])
        cam.select_poles()
        # frf_cam_plot.set_data(cam.freq, np.mean(np.abs(cam.frf), axis=0))
        plt.draw()
        plt.pause(0.001)
        input_string = input('Are the selected poles correct? (y/n): ')
        if input_string == 'y':
            break
        elif input_string == 'n':
            continue

    with open(file_path_cam, 'wb') as f:
        pkl.dump(cam, f)
    # cam_d.select_closest_poles(cam.nat_freq, f_window=5, fn_temp=0.001, xi_temp=0.05)
    # while True:
    #     cam_d.select_poles(approx_nat_freq=cam.nat_freq)
    #     frf_d_cam_plot.set_data(cam_d.freq, np.mean(np.abs(cam_d.frf), axis=0))
    #     plt.draw()
    #     plt.pause(0.001)
    #     input_string = input('Are the selected poles correct? (y/n): ')
    #     if input_string == 'y':
    #         break
    #     elif input_string == 'n':
    #         continue

    # with open(file_cam_d, 'wb') as f:
    #     pkl.dump(cam_d, f)


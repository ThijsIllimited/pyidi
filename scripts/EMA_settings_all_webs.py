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
for file in files:
    # Get the file information
    name_video = os.path.basename(file)
    # if re.search(pattern, name_video):
    #     name_video_base = re.sub(pattern, '', name_video)
    root_video = os.path.dirname(file)
    df_filtered = df_file_description[df_file_description['filename'].isin([name_video])]
    indices = df_filtered.index
    peak_n = df_filtered['peak_n'].item()
    nut_idx = df_filtered['nut_idx'].item()
    d_lim = df_filtered['d_lim'].item()
    if peak_n != 'nan':
        continue
    EMA_structure = EMA_Structure(name_video)
    EMA_structure.open_impact_data()
    comment = EMA_structure.impact_data['comment']
    if comment == '':
        peak_n = 0
    elif 'first' in comment:
        peak_n = 0
    elif 'second' in comment:
        peak_n = 1
    elif 'third' in comment:
        peak_n = 2
    else:
        peak_n = 0

    # Initialize signals
    EMA_structure.initialize_signals()
    first_zero_id_force = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=peak_n, treshold=0.05)
    peak_force = EMA_structure.F_peak

    fig, ax = plt.subplots()
    plt.ion()
    ax.set_title(f'Current peak_n: {peak_n}, comment: {comment}, peak_force: {peak_force}')
    ax.plot(EMA_structure.t_force_raw, EMA_structure.force_raw)
    start_plot = ax.plot(EMA_structure.t_force_raw[first_zero_id_force], EMA_structure.force_raw[first_zero_id_force], color='r*')
    plt.show()

    # User input
    while True:
        new_peak_n = input('New peak_n:..., q to quit')
        if new_peak_n == 'q':
            break
        new_peak_n = int(new_peak_n)
        first_zero_id_force_new = EMA_structure.find_signal_start(EMA_structure.force_raw, peak_n=new_peak_n, treshold=0.05)
        ax.set_title(f'Current peak_n: {new_peak_n}, comment: {comment}, peak_force: {peak_force}')
        start_plot[0].set_data(EMA_structure.t_force_raw[first_zero_id_force_new], EMA_structure.force_raw[first_zero_id_force_new])
        plt.draw()
        plt.pause(0.001)






    
    # video = EMA_structure.open_video(add_extension = False)
    # prey_ij = ast.literal_eval(df_filtered['prey_ij'].item())
    # EMA_structure.nut_idx((prey_ij[0], prey_ij[1]), exclude_high_amplitude = True, d_lim = d_lim)
    

# for file in files:
#     # import impact data
#     # plot impact data, title is current peak_n, and comment
#     # user input: new peak_n
#     # update plot until user input is 'q'


#     # import video
#     video = EMA_structure.open_video(add_extension = False)

#     # import ROI
#     # import displacement data
#     DIC_structure = DIC_Structure(file)
#     df = DIC_structure.list_test_data(test_range = range(1, 100), robostness_check = False)
#     last_row = df.iloc[-1]  # Get the last row of the DataFrame
#     test_number = last_row['test_number']  # Get the test number of the last row
#     # play video, plot ROI, and valid displacement data.  title is d_lim, smooth_lim, max_drift, max_end_drift
#     # user input: (new d_lim, smooth_lim, max_drift, max_end_drift)
#     # Replay video until user input is 'q'

#     # Save new settings
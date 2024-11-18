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

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)
# pattern = r'_S\d+\.cihx$'

fig1 = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1])

# Add subplots to the first column (4 rows)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[2:, 0])
ax5 = fig1.add_subplot(gs[:2, 1])
ax6 = fig1.add_subplot(gs[2:, 1])

plt.ion()
ax0_2 = ax1.twinx()
ax1_2 = ax2.twinx()

force_raw_plot = ax0_2.plot([], [],'b-', label='Force (raw))')
disp_nut_plot  = ax1.plot([], [],'r--', label=f'Displacement nut y')
disp_nut_damped_plot = ax2.plot([], [],'r--', label=f'Displacement nut y - damped')

ax1.set_ylabel('pixels (-)')
ax0_2.set_ylabel('Force (N)')
ax1.legend(loc='upper right')
ax2.set_ylabel('pixels (-)')
ax1_2.set_ylabel('Force (N)')

# set legend to north east
ax1_2.legend()
ax2.set_xlabel('Time (s)')

ax5.imshow(np.zeros((10,10)), cmap='gray', vmin=0, vmax=2**16-1)
ax5.set_xticks([])
ax5.set_yticks([])
smooth_points_plot = ax5.plot([],[], 'm.', markersize=3, label='Smooth')
drift_points_plot = ax5.plot([],[], 'r*', label='Drift general')
drift_points_plot2 = ax5.plot([],[], 'b.', label='Drift end', markersize=3)
ax5.legend()

H1_plot, = ax4.semilogy([], 50*[], linewidth=0.5, label='H1')


# ax6.imshow(np.zeros((10,10)), cmap='gray', vmin=0, vmax=2**16-1)

ax6.set_title('valid points')
im = ax6.imshow(np.zeros((512, 1024)), cmap='gray', vmin=0, vmax=2**16-1)
text = ax6.text(0.65, 0.05, '', transform=ax6.transAxes, color='black', ha='right', va='bottom')
pts, = ax6.plot([], [], 'r.', markersize=3)
ax6.set_xticks([])
ax6.set_yticks([])
# valid_points_plot = ax6.plot([],[], 'r.')


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
        prey_ij = ast.literal_eval(df_filtered['prey_ij'].item())
        spider_ij = ast.literal_eval(df_filtered['spider_ij'].item())
    except:
        print(f'spider and prey locations not set in {name_video}')
        invalid_files.append(name_video)
        continue
    try:
        peak_n = df_filtered['peak_n'].item()
        peak_F = df_filtered['peak_F'].item()
        peak_F_threshold = df_filtered['peak_F_threshold'].item()
    except:
        print(f'peak_n is not defined in {name_video} not found in file description (Impact_settings_all_webs.py)')
        invalid_files.append(name_video)
        continue
    try:
        shift = ast.literal_eval(df_filtered['shift'].item())
        d_lim = df_filtered['d_lim'].item()
        test_number = int(df_filtered['test_number'].item())
        nut_idx = int(df_filtered['nut_idx'].item())
    except:
        print(f'd_lim not set in {name_video} (EMA_settings_all_webs.py)')
    
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

    smooth_lim = df_filtered['smooth_lim'].item()
    # if isinstance(smooth_lim, str):
        # smooth_lim = eval(smooth_lim)
    if not np.isnan(smooth_lim):
        continue

    EMA_structure = EMA_Structure(name_video)
    # try:
    #     shift = ast.literal_eval(df_filtered['shift'].item())
    #     test_number = df_filtered['test_number'].item()
    #     test_number = int(test_number)
    #     nut_idx = int(nut_idx)
    #     nut_idx = df_filtered['nut_idx'].item()
    # except:
    #     shift = [0, 0]
    #     test_number = 1
    #     # nut_idx = EMA_structure.nearest_nut_index
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

    time_windows = np.linspace(0, 1, len(EMA_structure.t_camera))
    exponential_w = np.exp(np.log(0.1) * time_windows)
    
    smooth_lim = 0.8
    max_drift = d_lim/3
    max_end_drift = d_lim/3

    EMA_structure.set_freq_properties(padding_ratio=1)
    EMA_structure.get_transfer_function(direction='y')

    smooth_signals = np.max(np.abs(np.diff(np.linalg.norm(EMA_structure.d, axis=2))), axis = 1) < smooth_lim
    non_drifting = np.abs(np.mean(np.linalg.norm(EMA_structure.d[:,:-100], axis=2), axis=1) < max_drift)
    non_drifting2 = np.abs(np.linalg.norm(EMA_structure.d[:,-1], axis=1)) < max_end_drift

    EMA_structure.valid_tps = smooth_signals & non_drifting & non_drifting2 & EMA_structure.exclude_high_amplitude
    smooth_dlim = ~smooth_signals & EMA_structure.exclude_high_amplitude
    drift_d_lim = ~non_drifting & EMA_structure.exclude_high_amplitude
    drift_d_lim2 = ~non_drifting2 & EMA_structure.exclude_high_amplitude

    EMA_structure.displacements_x_damp = EMA_structure.displacements_x*exponential_w
    EMA_structure.displacements_y_damp = EMA_structure.displacements_y*exponential_w
    
    force_raw_plot[0].set_data(EMA_structure.t_force, EMA_structure.force)
    disp_nut_plot[0].set_data(EMA_structure.t_camera, EMA_structure.displacements_y[EMA_structure.nearest_nut_index,:])
    disp_nut_damped_plot[0].set_data(EMA_structure.t_camera, EMA_structure.displacements_y_damp[EMA_structure.nearest_nut_index,:])

    ax1.set_xlim([-0.02, EMA_structure.t_camera[-1]])
    ax1.set_ylim(-d_lim*1.1, d_lim*1.1)
    ax2.set_ylim(-d_lim*1.1, d_lim*1.1)
    ax2.set_xlim([-0.02, EMA_structure.t_camera[-1]])

    ax4.cla()
    for H1_i in EMA_structure.H1[EMA_structure.valid_tps]:
        ax4.semilogy(EMA_structure.freq_force, np.abs(H1_i), 'r', alpha=0.2, linewidth=0.2)
    ax4.set_xlim([0, 50])
    ax4.set_ylim([1e-1, 1e4])

    ax5.set_title(f'drift limit general: smooth limit: {smooth_lim}, {max_drift}, drift limit end: {max_end_drift}')
    ax5.imshow(video.reader.get_frame(0), cmap='gray', vmin=0, vmax=2**16-1)
    smooth_points_plot[0].set_data(EMA_structure.tp[smooth_dlim,1], EMA_structure.tp[smooth_dlim,0])
    drift_points_plot[0].set_data(EMA_structure.tp[drift_d_lim,1], EMA_structure.tp[drift_d_lim,0])
    drift_points_plot2[0].set_data(EMA_structure.tp[drift_d_lim2,1], EMA_structure.tp[drift_d_lim2,0])
    
    ax6.cla()
    im = ax6.imshow(np.zeros((512, 1024)), cmap='gray', vmin=0, vmax=2**16-1)
    text = ax6.text(0.65, 0.05, '', transform=ax6.transAxes, color='black', ha='right', va='bottom')
    pts, = ax6.plot([], [], 'r.', markersize=3)
    def update_frame(frame_idx):
        im.set_data(video.reader.get_frame(frame_idx))
        text.set_text(f'Frame: {frame_idx}')
        pts.set_data(td[EMA_structure.valid_tps,frame_idx, 1], td[EMA_structure.valid_tps,frame_idx, 0])
        return im, text, pts
    
    if 'ani' in locals():
        ani.event_source.stop()
    ani = animation.FuncAnimation(fig1, update_frame, frames=range(200, video.reader.N-1, 5), interval=30, blit=True)
    max_d = np.max(np.abs(EMA_structure.displacements_raw[EMA_structure.nearest_nut_index,:,0]))
    ax1.set_title(f'{name_video} \n Current d_lim: {d_lim}\n max d: {max_d} \n comment: {comment}')
    plt.show()

    while True:
        input_string = input(f"Enter new (smooth_lim, max_drift, max_end_drift), or 'q' to quit: ")
        if input_string == 'q':
            break
        try:
            values = input_string.split(',')
            smooth_lim = float(values[0])
            max_drift = float(values[1])
            max_end_drift = float(values[2])
        except:
            print('Invalid input')
            continue
        smooth_signals = np.max(np.abs(np.diff(np.linalg.norm(EMA_structure.d, axis=2))), axis = 1) < smooth_lim
        non_drifting = np.abs(np.mean(np.linalg.norm(EMA_structure.d[:,:-100], axis=2), axis=1) < max_drift)
        non_drifting2 = np.abs(np.linalg.norm(EMA_structure.d[:,-1], axis=1)) < max_end_drift

        EMA_structure.valid_tps = smooth_signals & non_drifting & non_drifting2 & EMA_structure.exclude_high_amplitude
        smooth_dlim = ~smooth_signals & EMA_structure.exclude_high_amplitude
        drift_d_lim = ~non_drifting & EMA_structure.exclude_high_amplitude
        drift_d_lim2 = ~non_drifting2 & EMA_structure.exclude_high_amplitude

        ax5.set_title(f'drift limit general: smooth limit: {smooth_lim}, {max_drift}, drift limit end: {max_end_drift}')
        smooth_points_plot[0].set_data(EMA_structure.tp[smooth_dlim,1], EMA_structure.tp[smooth_dlim,0])
        drift_points_plot[0].set_data(EMA_structure.tp[drift_d_lim,1], EMA_structure.tp[drift_d_lim,0])
        drift_points_plot2[0].set_data(EMA_structure.tp[drift_d_lim2,1], EMA_structure.tp[drift_d_lim2,0])
        
        ax4.cla()
        for H1_i in EMA_structure.H1[EMA_structure.valid_tps]:
            ax4.semilogy(EMA_structure.freq_force, np.abs(H1_i), 'r', alpha=0.2, linewidth=0.2)
        ax4.set_xlim([0, 50])
        ax4.set_ylim([1e-1, 1e4])

        ax6.cla()
        im = ax6.imshow(np.zeros((512, 1024)), cmap='gray', vmin=0, vmax=2**16-1)
        text = ax6.text(0.65, 0.05, '', transform=ax6.transAxes, color='black', ha='right', va='bottom')
        pts, = ax6.plot([], [], 'r.', markersize=3)
        if 'ani' in locals():
            ani.event_source.stop()
        ani = animation.FuncAnimation(fig1, update_frame, frames=range(200, video.reader.N-1, 5), interval=30, blit=True)
    df_file_description.loc[indices, 'smooth_lim'] = smooth_lim
    df_file_description.loc[indices, 'max_drift'] = max_drift
    df_file_description.loc[indices, 'max_end_drift'] = max_end_drift
    print(f'File {file_i} of {len(files)} saved')
    df_file_description.to_csv(file_path_settings)
print(invalid_files)
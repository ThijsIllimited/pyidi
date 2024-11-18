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
df_file_description = pd.read_csv('I:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
df_file_description = df_file_description.loc[:, ~df_file_description.columns.str.startswith('Unnamed')]

# List all files
files = glob.glob('D:/thijsmas/HSC/**/*.cihx', recursive=True)

D_in = 40
D_out = 100
nut_wh = (15, 60)
n_modes = 8
n_peak_modes = 4

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

    file_path_cam = os.path.join(root_video, f'{name_video}_cam_2.pkl')
    with open(file_path_cam, 'rb') as file:
        cam = pkl.load(file)
    pol_order = len(cam.all_poles)
    
    participation_factors = cam.partfactors
    # part_real = []
    part_abs = []
    for i in range(len(cam.nat_freq)):
        # part_real.append(np.abs(np.real(cam.partfactors[cam.pole_ind[i][0]][cam.pole_ind[i][1]])))
        part_abs.append(np.abs(cam.partfactors[cam.pole_ind[i][0]][cam.pole_ind[i][1]]))
    freq_args = np.argmin(np.abs(cam.freq - np.array(cam.nat_freq)[:, np.newaxis]), axis=1)
    Acquisition_period  = EMA_structure.t_camera[-1]
    S_xx = 1/Acquisition_period * np.conj(EMA_structure.disp_fft_y) * EMA_structure.disp_fft_y
    mean_S_xx = np.mean(S_xx[EMA_structure.valid_tps], axis=0)
    S_xx_peaks = np.abs(mean_S_xx[freq_args])
    highest_fns = np.argsort(S_xx_peaks)[::-1]
    if cam.nat_freq[0] == 0.0:
        highest_fns = [fn for fn in highest_fns if fn != 0]
    # order_real = np.argsort(part_real)[::-1]
    order_abs = np.argsort(part_abs)[::-1]
    if cam.nat_freq[0] == 0.0:
        part_mode_selection = sorted(order_abs[:n_modes+1])[1:]
    else:
        part_mode_selection = sorted(order_abs[:n_modes])
    peak_modes = list(highest_fns[:n_peak_modes])
    remaining_modes = [mode for mode in part_mode_selection if mode not in peak_modes]
    mode_selection = sorted(peak_modes + remaining_modes[:n_modes - n_peak_modes])

    name_video_base = name_video.split('.')[0]
    file_name_dist = os.path.join(root_video, f"{name_video_base}_dist.pkl")
    with open(file_name_dist, 'rb') as f:
        dist = pkl.load(f)
    homography_matrix = dist["homography_matrix"]
    video = EMA_structure.open_video(add_extension = False)
    still_image = video.reader.get_frame(0)
    height, width = still_image.shape
    undistorted_image = cv2.warpPerspective(still_image, homography_matrix, (width, height))
    zero_columns = np.all(undistorted_image == 0, axis=0)
    first_true = np.argmax(~zero_columns)  # First True
    last_true = len(zero_columns) - np.argmax(~zero_columns[::-1]) - 1

    tp = EMA_structure.tp[EMA_structure.valid_tps]
    tp = np.dot(homography_matrix, np.array([tp[:, 1], tp[:, 0], np.ones(tp.shape[0])]))
    tp = (tp[1::-1]/tp[2][np.newaxis, :]).T

    fig = plt.figure(figsize=(21, 16)) 
    gs = gridspec.GridSpec(5, n_modes, height_ratios=[.6, .6, .6, .6, 1]) #height_ratios=[.5, 1, 1, 1],
    ax_real = [fig.add_subplot(gs[0, i]) for i in range(n_modes)]
    ax_imag = [fig.add_subplot(gs[1, i]) for i in range(n_modes)]
    ax_real_hub = [fig.add_subplot(gs[2, i]) for i in range(n_modes)]
    ax_imag_hub = [fig.add_subplot(gs[3, i]) for i in range(n_modes)]
    ax_FRF =  fig.add_subplot(gs[4, :2])
    ax_Sxx  = fig.add_subplot(gs[4, 2:4])
    ax_MAC  = fig.add_subplot(gs[4, 4:6])
    ax_info = fig.add_subplot(gs[4, 6:])

    mean_frf = np.mean(np.abs(cam.frf), axis=0)
    ax_FRF.semilogy(cam.freq, mean_frf, '-')
    for mode, mode_f in zip(mode_selection, np.array(cam.nat_freq)[mode_selection]):
        ax_FRF.vlines(mode_f, ymin=1e0, ymax=1e4, color='g', linestyle='--')
        ax_FRF.text(mode_f, 1e4, f"{mode+1:}", ha='center', va='bottom')
    ax_FRF.set_xlabel('Frequency [Hz]')
    ax_FRF.set_ylabel('FRF (receptance) [pixel/N]')

    fig, ax_MAC = plot_MAC(cam, fig, ax_MAC, mode_vec = mode_selection)
    
    arg_higer = np.argmin(np.abs(EMA_structure.freq_camera - cam.upper))
    S_xx_min, S_xx_max = np.min(mean_S_xx[:arg_higer]), np.max(mean_S_xx[:arg_higer])
    ax_Sxx.semilogy(EMA_structure.freq_camera, mean_S_xx, '-', lw = 2)
    ax_Sxx.set_xlim([0, cam.upper])
    ax_Sxx.set_ylim([S_xx_min*0.1, S_xx_max*1.3])
    ax_Sxx.set_title('S_xx [pixel^2]')
    ax_Sxx.set_xlabel('Frequency [Hz]')
    for mode, mode_f in zip(mode_selection, np.array(cam.nat_freq)[mode_selection]):
        ax_Sxx.vlines(mode_f, ymin=S_xx_min, ymax=S_xx_max, color='g', linestyle='--')  
        ax_Sxx.text(mode_f, S_xx_max, f"{mode+1:}", ha='center', va='bottom')

    nat_freq = np.array(cam.nat_freq)[mode_selection]
    nat_freq_ratios = nat_freq/nat_freq[0]
    nat_freq_arg = np.argmin(np.abs(EMA_structure.freq_camera[:, np.newaxis] - nat_freq), axis=0)
    Sxx_peaks = np.abs(mean_S_xx[nat_freq_arg])
    Sxx_peaks_norm = Sxx_peaks/Sxx_peaks[0]
    ax_Sxx.scatter(nat_freq, Sxx_peaks, color='m', s=40, marker='x')

    Sxx_peaks_norm_str = '\n'.join([f"mode {mode+1:<2}: {peak:<5.2g}" for mode, peak in zip(mode_selection, Sxx_peaks_norm)])
    nat_freq_ratios_str = '\n'.join([f"mode {mode+1:<2}: {ratio:<5.2f}" for mode, ratio in zip(mode_selection, nat_freq_ratios)])

    info_text0 = f"""
        Figure Information:
Peak Force:  {EMA_structure.peak_F:.2f} N
Double Tap:  {EMA_structure.double_tap}
Prey Dis:    {np.linalg.norm([EMA_structure.prey_ij_d[0]-EMA_structure.spider_ij_d[0], EMA_structure.prey_ij_d[1] - EMA_structure.spider_ij_d[1]]):.2f} px
$S_{{xx}}$ ratios (px²/px²):\n {Sxx_peaks_norm_str}
    """

    info_text01 = f"""\n
Lower Bound: {cam.lower} Hz
Upper Bound: {cam.upper} Hz
Poly. Order: {pol_order}
$f_n$ ratios (Hz/Hz):\n {nat_freq_ratios_str}
    """

    ax_info.text(0.01, 1, info_text0, transform=ax_info.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    ax_info.text(.99, 1, info_text01, transform=ax_info.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
    ax_info.axis('off')
    fig.suptitle(EMA_structure.file_name)

    for ax_real, ax_imag, mode_n in zip(ax_real, ax_imag, mode_selection):
        ax_real.imshow(undistorted_image, cmap = 'gray')
        ax_real.set_xlim([first_true, last_true])
        ax_imag.imshow(undistorted_image, cmap = 'gray')
        ax_imag.set_xlim([first_true, last_true])
        fig, ax_real, ax_imag = plot_mode_shape_flat(cam, fig, ax_real, ax_imag, mode_n, tp, mask = None)
        ax_real.scatter([EMA_structure.spider_ij_d[0], EMA_structure.prey_ij_d[0]], [EMA_structure.spider_ij_d[1], EMA_structure.prey_ij_d[1]], color='g', s=20, marker='x')
        ax_imag.scatter([EMA_structure.spider_ij_d[0], EMA_structure.prey_ij_d[0]], [EMA_structure.spider_ij_d[1], EMA_structure.prey_ij_d[1]], color='g', s=30, marker='x')
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_imag.set_xticks([])
        ax_imag.set_yticks([])
    
    for fig_i, (ax_real_hub, ax_imag_hub, mode_n) in enumerate(zip(ax_real_hub, ax_imag_hub, mode_selection)):
        fn_arg = nat_freq_arg[fig_i]
        EMA_structure.plot_hub_modes(cam, fig, ax_real_hub, ax_imag_hub, mode_n, video, D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = True, normalize_colors = True)
        EMA_structure.plot_Sxx_sections(fig, ax_imag_hub, ax_real_hub, S_xx[:,fn_arg], n_sections = 8, D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = True)
        ax_real_hub.set_xticks([])
        ax_real_hub.set_yticks([])
        ax_imag_hub.set_xticks([])
        ax_imag_hub.set_yticks([])
    
    EMA_structure.plot_Sxx_freq(ax_Sxx, S_xx)

    fig.suptitle(EMA_structure.file_name)
    # plt.show()
    fig.savefig(f'{root_video}/{name_video}_main_modes.png')
    print(f'Figure for {name_video} is generated')
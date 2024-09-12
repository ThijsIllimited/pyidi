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

# Import test data
df_file_description = pd.read_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA.csv')
# Back up the data
df_file_description.to_csv('H:/My Drive/PHD/HSC/file_descriptions_wEMA_backup.csv')

# List all files
files_cam = glob.glob('D:/thijsmas/HSC/**/*_cam.pkl', recursive=True)
files_ema = glob.glob('D:/thijsmas/HSC/**/*_EMA.pkl', recursive=True)
files_cam_d = glob.glob('D:/thijsmas/HSC/**/*_cam_damped.pkl', recursive=True)
files_ema_d = glob.glob('D:/thijsmas/HSC/**/*_EMA_damped.pkl', recursive=True)

# fig_poles, ax_poles = plt.subplots()
# pole_plot, = ax_poles.plot([], [], 'x')


# fig, ax = plt.subplots(2, 1)
# plt.ion()
# H_plot ,            = ax[0].semilogy([], [], 'k-', label='H')
# frf_cam_plot ,      = ax[0].semilogy([], [], 'r--', label='frf - cam')
# # poles_cam_plot ,    = ax[0].plot([], [], 'x', label='poles - cam')
# H_d_plot ,          = ax[1].semilogy([], [], 'k-', label='H damped')
# frf_d_raw_plot ,    = ax[1].semilogy([], [], 'r--', label='frf damped - raw')
# frf_d_cam_plot ,    = ax[1].semilogy([], [], 'r--', label='frf damped - cam')
# poles_d_cam_plt ,   = ax[1].plot([], [], 'x', label='poles damped - cam')
# ax[0].set_xlim([0, 300])
# ax[1].set_xlim([0, 300])
# ax[0].set_ylim([1e-1, 1e4])
# ax[1].set_ylim([1e-1, 1e4])


# fig2, ax2 = plt.subplots()

for file_cam, file_cam_d, file_ema, file_ema_d in zip(files_cam, files_cam_d, files_ema, files_ema_d):
    # Load the files
    with open(file_cam, 'rb') as f:
        cam = pkl.load(f)
    # with open(file_cam_d, 'rb') as f:
    #     cam_d = pkl.load(f)
    if 'nat_freq' in cam.__dict__.keys():
        continue
    with open(file_ema, 'rb') as f:
        EMA_structure = pkl.load(f)
    with open(file_ema_d, 'rb') as f:
        EMA_structure_d = pkl.load(f)
    # ax2.semilogy(EMA_structure_d.freq_camera, np.mean(np.abs(EMA_structure_d.H1[EMA_structure_d.valid_tps]), axis=0))   
    # plt.show()
    mean_abs_H1 = np.mean(np.abs(EMA_structure_d.H1[EMA_structure_d.valid_tps]), axis=0)
    max_amp = np.max(mean_abs_H1)
    # H_plot.set_data(EMA_structure.freq_camera, np.mean(np.abs(EMA_structure.H1[EMA_structure.valid_tps]), axis=0))
    # H_d_plot.set_data(EMA_structure_d.freq_camera, np.mean(np.abs(EMA_structure_d.H1[EMA_structure_d.valid_tps]), axis=0))
    peaks, _ = find_peaks(mean_abs_H1[EMA_structure_d.freq_camera<300], distance=6, width=3)
    # poles_d_cam_plt.set_data(EMA_structure_d.freq_camera[peaks], mean_abs_H1[peaks])
    try:
        file_name = file_cam.split('\\')[-1].split('.cihx_cam.pkl')[0]
    except:
        file_name = file_cam.split('\\')[-1].split('_cam.pkl')[0]
    # frf_d_raw_plot.set_data(cam_d.freq, np.mean(np.abs(cam_d.frf), axis=0))
    plt.draw()
    plt.pause(0.001)
    while True:
        cam.select_poles(approx_nat_freq=EMA_structure_d.freq_camera[peaks])
        # frf_cam_plot.set_data(cam.freq, np.mean(np.abs(cam.frf), axis=0))
        plt.draw()
        plt.pause(0.001)
        input_string = input('Are the selected poles correct? (y/n): ')
        if input_string == 'y':
            break
        elif input_string == 'n':
            continue

    with open(file_cam, 'wb') as f:
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


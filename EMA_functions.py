import pyidi
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import find_peaks
import warnings as warn
from scipy import stats

def open_video(file_name, paths_to_check=['H:/My Drive/PHD/HSC', 'D:/HSC', 'F:/', 'E:/thijs/', 'C:/Users/thijs/Documents/HSC/', 'D:/thijsmas/HSC']):
    """
    Opens a video file based on the given file name and paths to check.

    Args:
        file_name (str): The name of the video file.
        paths_to_check (list): A list of paths to check for the video file.

    Returns:
        pyidi.pyIDI: The pyIDI object representing the opened video file.
    """
    file_name_video = file_name + "_S01.cihx"
    for root, dirs, files in os.walk(paths_to_check, topdown=False):
        if file_name_video in files:
            file_path = os.path.join(root, file_name_video)
            print(file_path)
            break
    return pyidi.pyIDI(file_path)

def plot_still_frame(video, sequential_image_n):
    """
    Plots a still frame from the given video object.

    Args:
        video (pyidi.pyIDI): The pyIDI object representing the video.
        sequential_image_n (int): The index of the sequential image to plot.
    """
    still_image = video.mraw[sequential_image_n]
    fig, ax = plt.subplots()
    ax.imshow(still_image, cmap='gray')
    plt.show()

def gradient_check_LK(ref_imgs, bit_depth=16, lim=60000, scale=0.3):
    fig, ax = plt.subplots(ncols=3, nrows=len(ref_imgs), figsize=(10, 10))
    fig.tight_layout()
    for ax_i, img in enumerate(ref_imgs):
        ax[ax_i,0 ].imshow(img, cmap='gray', vmin=0, vmax=2**bit_depth-1)
    X, Y = np.meshgrid(np.arange(0, ref_imgs[0].shape[1]), np.arange(0, ref_imgs[0].shape[0]))
    for ax_i, img in enumerate(ref_imgs):
        ax[ax_i,1].set_aspect('equal')
        ax[ax_i,1].set_ylim(img.shape[0]-.5, -.5)
        ax[ax_i,1].set_xlim(-.5,img.shape[0]-.5)
        grad = np.gradient(img)

        ax[ax_i,1 ].quiver(X, Y, grad[1], -grad[0], scale=300000)
        At   = np.array([grad[1].flatten(), grad[0].flatten()])
        A    = At.T
        eig_vals, eig_vecs = np.linalg.eigh(A.T @ A)
        print(np.linalg.inv(A.T @ A))
        print(eig_vals)
        ax[ax_i,2].plot(grad[1], grad[0],'k.')
        ax[ax_i,2].plot([0, -scale*eig_vals[0]**0.5*eig_vecs[0,0]], [0, -scale*eig_vals[0]**0.5*eig_vecs[0,1]],'r')
        ax[ax_i,2].plot([0, -scale*eig_vals[1]**0.5*eig_vecs[1,0]], [0, -scale*eig_vals[1]**0.5*eig_vecs[1,1]],'r')
        ax[ax_i,2].set_aspect('equal')
        ax[ax_i,2].set_xlim(-lim, lim)
        ax[ax_i,2].set_ylim(lim, -lim)
        ax[ax_i,2].set_xlabel('Ix')
        ax[ax_i,2].set_ylabel('Iy')
    plt.show()

def initialize_signals(impact_data, approx_peak_heigth = 1.5):
    """
    Prepares the force and accelerations signal for the impact data.
    Args:
        impact_data (dict): The impact data.
        approx_peak_heigth (float): The approximate peak height of the force signal.
    """
    fs_force = impact_data['NI']['sample_rate']
    force_index = impact_data['NI']['channel_names'].index('Force')
    force = impact_data['NI']['data'][:,force_index]
    start_index = np.where(impact_data['NI']['time'] >= approx_peak_heigth)[0][0]
    impact_mean = np.mean(force[start_index+2:start_index+1000])
    force -= impact_mean


    accel_index = impact_data['NI']['channel_names'].index('Acceleration')
    accel = impact_data['NI']['data'][:,accel_index]
    mean_accel = np.mean(accel[start_index+2:start_index+1000])
    accel = -(accel-mean_accel)/9.81
    return fs_force, force, accel

def initialize_displacement(displacements, idx='all', dir='y'):
    """
    Prepares the displacement signal for the impact data.
    Args:
        displacements (dict): The displacement data from all tracking dots.
        idx (int): The index of the displacement signal to use.
        dir (str): The direction of the displacement signal to use.
    """
    if idx == 'all' and dir == 'y':
        return displacements['displacement'][:, :, 0]
    elif idx == 'all' and dir == 'x':
        return displacements['displacement'][:, :, 1]

    if dir == 'y':
        return displacements['displacement'][:, idx, 0, np.newaxis] if len(idx) > 0 else displacements['displacement'][:, :, 0]
    elif dir == 'x':
        return displacements['displacement'][:, idx, 1, np.newaxis] if len(idx) > 0 else displacements['displacement'][:, :, 1]

def prepare_signals(force, disp, fs_camera, fs_force, t_camera, t_force, peak_id = 0, threshold  = 0.001, approximate_force_peak_height = 0.5):
    """
    Prepare the signals
    - Allign impact with the point where nut starts moving
    - Add n zeros before signals against leakage
    - clip signals such that they end at exact same time point
    - Zero all the force except the impact
    Args:
        force (np.array): The force signal.
        disp (np.array): The displacement signal.
        fs_camera (float): The sampling frequency of the camera.
        fs_force (float): The sampling frequency of the force sensor.
        t_camera (np.array): The time signal of the camera.
        t_force (np.array): The time signal of the force sensor.
        peak_id (int): The index of the peak in the displacement signal to use.
        threshold (float): Value concidered to be near zero to use for the force and acceleration signal.
        approximate_force_peak_height (float): The approximate peak height of the force signal.

    """
    frames_per_gcd      = int(fs_camera / math.gcd(int(fs_camera), int(fs_force)))
    n_zeros_before_cam  = frames_per_gcd
    n_zeros_before_for  = int(n_zeros_before_cam * fs_force / fs_camera)
    disp_peaks, _       = find_peaks(disp[peak_id], height=0.5, distance=fs_camera*0.01)
    near_zero_indices_cam   = np.where(disp[:disp_peaks[0]] < threshold)[0]
    first_cam_near_zero_index = near_zero_indices_cam[-1]
    # Shift the time signal such that the impact - n_zeros_before_cam is at t=0, 
    t_camera2           = np.round(t_camera - t_camera[first_cam_near_zero_index-n_zeros_before_cam], 7)
    # Clip the time
    time_cam_clipped    = t_camera2[first_cam_near_zero_index-n_zeros_before_cam:]
    disp_clipped        = disp[:, first_cam_near_zero_index-n_zeros_before_cam:]
    disp_clipped[:n_zeros_before_cam] = 0
    force_peaks, _  = find_peaks(force, height=approximate_force_peak_height)
    near_zero_indices_for   = np.where(force[:force_peaks[0]] < threshold)[0]
    first_for_near_zero_index = near_zero_indices_for[-1]
    near_zero_indices_for_after   = (np.where(force[force_peaks[0]+1:force_peaks[0]+10] < threshold)[0]) + force_peaks[0]+1
    first_for_near_zero_index_after = near_zero_indices_for[0]
    t_force             = np.round(t_force - t_force[first_for_near_zero_index - n_zeros_before_for],7)
    t_force_index_end   = np.argmin(np.abs(t_force - t_camera2[-1]))

    time_force_clipped    = t_force[first_for_near_zero_index- n_zeros_before_for:t_force_index_end]
    force_clipped       = force[first_for_near_zero_index- n_zeros_before_for:t_force_index_end]

    # Cut off signals such that they end at the same time. Loop is because sample ratios 
    for i in range(1, frames_per_gcd+2): 
        if time_cam_clipped[-i] in time_force_clipped:
            index = np.where(time_cam_clipped[-i] == time_force_clipped)[0][0]
            time_cam_clipped = time_cam_clipped[:-i+1]
            disp_nut_clipped = disp_nut_clipped[:-i+1]
            disp_clipped = disp_clipped[:, :-i+1]
            time_force_clipped = time_force_clipped[:index+1]
            force_clipped = force_clipped[:index+1]
            break
    if time_cam_clipped[-1] != time_force_clipped[-1]:
        warn.warn('Time signals do not end at the same time. This is probably due to a rounding error. The last sample of the force signal is removed.')

    # Zero the force signal around the impact (The rest is all noise)
    force_clipped_mod = np.copy(force_clipped)
    force_clipped_mod[force_clipped_mod < 0.1] = 0
    return time_cam_clipped, disp_clipped, time_force_clipped, force_clipped_mod

def define_frequency_spectrum(signal, fs, extension_ratio = 1):
    """
    Defines the frequency spectrum for the given signal.
    Args:
        signal (np.array): The signal to define the frequency spectrum for.
        fs (float): The sampling frequency of the signal.
        extension_ratio (float): The ratio to extend the signal with.
    """
    n = len(signal)
    n = int(np.ceil(n*extension_ratio))
    return np.fft.rfftfreq(n, 1/fs)

def FFT_on_signal(signal, n_d = None):
    if n_d is None:
        n_d = len(signal)
    return np.fft.rfft(signal, n=n_d) / n_d

def get_transfer_function(Disp, Force, Acquisition_period, type = 'H1'):
    """
    Find transfer function between force and displacement
    Args:
        Disp (np.array): The displacement signal.
        Force (np.array): The force signal.
        Acquisition_period (float): The acquisition period of the signals.
        type (str): The type of transfer function to calculate.
    """
    S_ff = 1/Acquisition_period * np.conj(Force) * Force
    S_fx = 1/Acquisition_period * np.conj(Force) * Disp
    S_xx = 1/Acquisition_period * np.conj(Disp) * Disp
    S_fx = 1/Acquisition_period * np.conj(Force) * Disp
    if type == 'H1':
        return S_fx / S_ff
    elif type == 'H2':
        return S_fx / S_xx
    elif type == 'Hv':
        return (S_fx / S_ff * S_fx / S_xx)**0.5
    
def H_outliers(H, z_limit = 1.):
    """
    Removes outliers from the transfer function.
    Args:
        H (np.array): The transfer function.
        z_limit (float): The threshold to use to remove outliers.
    """
    z_scores = stats.zscore(H, axis=0)
    z_scores = np.mean(np.abs(z_scores), axis=1)
    biggest_outliers = np.where(z_scores > z_limit)[0]
    return biggest_outliers

def find_valid_points(cam, max_amplitude=5, exlude_outliers=True, z_limit=1.):
    valid_ids = np.all(np.abs(cam.A) <= max_amplitude, axis=1)
    not_nan_ids = ~np.isnan(cam.A, axis=1)
    valid_ids = valid_ids & not_nan_ids
    if exlude_outliers:
        valid_ids = valid_ids & ~H_outliers(cam.H, z_limit)
    return valid_ids

def plot_FRF(cam):
    H, A = cam.get_constants(whose_poles='own', FRF_ind='all', least_squares_type='new')
    fig, ax = plt.subplots(2,1,figsize=(15, 9))
    y_cor_anot_lin = [3500, 1000, 2500, 1000, 1500, 500, 1000,1500,500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000,1500,500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000]
    ax[0].plot(cam.freq, np.average(np.abs(H), axis=0))
    for i, freq in enumerate(cam.nat_freq):
        ax[0].axvline(x=freq, color='r', linestyle='--')
        ax[0].annotate(f"Mode {i+1}\n{freq:.2f} Hz", xy=(freq, y_cor_anot_lin[i%len(y_cor_anot_lin)]), xytext=(freq, y_cor_anot_lin[i%len(y_cor_anot_lin)]), ha='center', fontsize=6)
    ax[0].set_ylabel('H')
    ax[1].semilogy(cam.freq, np.average(np.abs(H), axis=0))
    for i, freq in enumerate(cam.nat_freq):
        ax[1].axvline(x=freq, color='r', linestyle='--')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('H')
    plt.show()

def plot_MAC(cam, n_modes = 15, mode_vec = None, cmap='plasma'):
    MAC = cam.autoMAC()
    if mode_vec is not None:
        MAC_copy = np.copy(MAC)
        MAC_copy = MAC[mode_vec, :]
        MAC_copy = MAC[:, mode_vec]
        MAC = MAC_copy
    fig, ax = plt.subplots( figsize=(8, 8))
    im = ax.imshow(MAC, cmap=cmap)
    fig.colorbar(im, ax=ax)
    n_modes = min(n_modes, len(cam.nat_freq))
    ax.set_xlim([0-.5, n_modes-.5])
    ax.set_ylim([n_modes-.5, 0-.5])
    plt.show()
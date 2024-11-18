import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import find_peaks
import warnings as warn
from scipy import stats
import pickle as pkl
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.animation as animation
import re
import matplotlib.cm as cm
import matplotlib.colors as col
import cv2
import matplotlib.patches as patches
import ast

class EMA_Structure:
    def __init__(self, file_name):
        self.file_name = file_name
        pattern = r'_S\d+\.cihx$'
        if re.search(pattern, self.file_name):
            self.file_name_base = re.sub(pattern, '', self.file_name)
        else:
            self.file_name_base = self.file_name
        self.d = None
        self.paths_to_check = [r'D:/HSC', r'F:/', r'E:/thijs/', r'C:/Users/thijs/Documents/HSC/', r'D:/thijsmas/HSC',r'D:/thijsmas/HSC - Ladisk', r'D:/thijsmas', r'I:/My Drive/PHD/HSC',]
        self.root_impact    = os.path.normpath(r'I:/My Drive/PHD/Data')
        self.root_simulations = r'G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations'
        self.root_disp      = r"C:/Users/thijsmas/Documents/GitHub/pyidi_data/displacements" #r"G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/displacements"
        self.root_cam       = r'G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/EMA models'
        self.path_EMA       = os.path.join(self.root_cam, f'{self.file_name}_cam.pkl')
        self.root_EMA_struct = r"C:/Users/thijsmas/Documents/GitHub/pyidi_data/EMA structure"#r'G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/EMA structure'
        pass

    def custom_paths(self, paths_to_check):
        self.paths_to_check = paths_to_check

    def custom_root_impact(self, root_impact):
        self.root_impact = root_impact

    def custom_root_disp(self, root_disp):
        self.root_disp = root_disp
        
    def open_video(self, add_extension = True):
        """
        Opens a video file based on the given file name and paths to check.

        Args:
            file_name (str): The name of the video file.
            paths_to_check (list): A list of paths to check for the video file.

        Returns:
            pyidi.pyIDI: The pyIDI object representing the opened video file.
        """
        import pyidi
        self.file_path = None
        if add_extension:
            self.file_name_video = self.file_name + "_S01.cihx"
        else:
            self.file_name_video = self.file_name
        for folder_path in self.paths_to_check:
            for root, dirs, files in os.walk(folder_path, topdown=False):
                if self.file_name_video in files:
                    self.file_root = root
                    self.file_path = os.path.join(root, self.file_name_video)
                    print(self.file_path)
                    break
            if self.file_path is not None:
                break
        if self.file_path is not None:
            video = pyidi.pyIDI(self.file_path)
            try:
                self.fs_camera = video.info['Record Rate(fps)']
                self.t_camera_raw = np.arange(video.info['Total Frame']) / self.fs_camera
            except:
                warn.warn('No camera frame rate found in the video file. Check the file name and paths to check.')
                    
            return video
        else:
            warn.warn('Video file not found. Check the file name and paths to check.')
            return 
    
    def open_video_compressed(self, file_name, file_root, frame_rate = 1000):
        import cv2 
        file_path = os.path.join(file_root, file_name)
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        all_frames = []
        while ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_frames.append(gray_frame[np.newaxis, :, :])
            ret, frame = cap.read()

        all_frames = np.concatenate(all_frames, axis=0) 
        cap.release()

        video = pyidi.pyIDI(all_frames) #(n time points, image height, image width)
        video.N = np.shape(all_frames)[0]
        video.info['Total Frame'] = video.N
        video.info['Frame Rate'] = frame_rate
        video.info['Image Width'] = np.shape(all_frames)[2]
        video.info['Image Height'] = np.shape(all_frames)[1]
        video.image_width = video.info['Image Width']
        video.image_height = video.info['Image Height']
        return video

    def open_impact_data(self):
        files = os.listdir(self.root_impact)
        for file in files:
            if self.file_name_base in file[16:-4]:
                path = os.path.join(self.root_impact, file)
                with open(path, 'rb') as f:
                    self.impact_data = pkl.load(f)
                break
        else:
            print(f"{self.file_name_base} is not in files")
            return

    def open_displacements(self, roi_size=None, reference_image=None, auto_nut_idx=True):
        self.file_name_displacement = self.file_name + '_d'
        if roi_size is not None:
            self.file_name_displacement = self.file_name_displacement + f'_rs{str(roi_size)}_ri{str(reference_image)}'
        extension = '.pkl'
        # Open displacement data
        self.root_disp      = r"G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/displacements"
        file_path = os.path.join(self.root_disp, self.file_name_displacement + extension)
        with open(file_path, 'rb') as f:
            # load the pickle data
            displacement_data = pkl.load(f)
        self.d = displacement_data['displacement']
        self.tp = displacement_data['tracking points']
        df = pd.read_csv('H:/My Drive/PHD/HSC/file_descriptions_wlocs.csv')
        self.properties = df[df['filename'] == self.file_name_video]
        self.spider_ij = eval(self.properties['spider_ij'].iloc[0])
        self.prey_ij = eval(self.properties['prey_ij'].iloc[0])
        if auto_nut_idx:
            self.nut_idx(self.prey_ij)
        return
    
    def open_cam_EMA(self):
        self.root_cam       = r'G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/EMA models'
        self.path_EMA       = os.path.join(self.root_cam, f'{self.file_name}_cam.pkl')
        with open(self.path_EMA, 'rb') as f:
            self.cam = pkl.load(f)

    def plot_still_frame(self, video, sequential_image_n, show_saturation = False, bit_depth = 16, tp_nut = False, tp = False, valid_only = False , labels = None):
        """
        Plots a still frame from the given video object.

        Args:
            video (pyidi.pyIDI): The pyIDI object representing the video.
            sequential_image_n (int or tuple): The index or range of sequential images to plot.
        """
        markers = ['o', 'v', '^', '<', '>', '.', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 's']
        if isinstance(sequential_image_n, tuple):
            accumulator = None
            frame_count = sequential_image_n[1] - sequential_image_n[0]

            # Loop through the specified frame range
            for i in range(sequential_image_n[0], sequential_image_n[1]):
                frame = video.reader.get_frame(i)
                if accumulator is None:
                    # Initialize the accumulator with the first frame
                    accumulator = np.zeros_like(frame, dtype=np.float64)
                accumulator += frame

            # Compute the mean image
            self.mean_image = accumulator / frame_count
        else:
            still_image = video.reader.get_frame(sequential_image_n)
        fig, ax = plt.subplots(figsize=(18, 8))
        fig.tight_layout()
        ax.imshow(still_image, cmap='gray', vmin=0, vmax=2**bit_depth-1)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_saturation:
            white_indices = np.where(still_image >= int(0.99*(2**bit_depth-1)))
            black_indices = np.where(still_image <= int(0.01*(2**bit_depth-1)))
            ax.plot(white_indices[1], white_indices[0], 'b.', alpha=0.2)
            ax.plot(black_indices[1], black_indices[0], 'g.', alpha=0.2)
        if tp_nut:
            ax.plot(self.tp[self.nearest_nut_index,1], self.tp[self.nearest_nut_index,0], 'r*')
        if isinstance(tp, np.ndarray):
            if tp.ndim == 2:
                ax.plot(tp[:, 1], tp[:,0], 'r*')
            elif tp.ndim == 3:
                for i, row in enumerate(tp):
                    ax.plot(row[:, 1], row[:, 0], label = labels[i], linestyle = 'None', marker = markers[i % len(markers)])
                ax.legend()
            else:
                if tp:
                    if valid_only:
                        ax.plot(self.tp[self.exclude_high_amplitude,1], self.tp[self.exclude_high_amplitude,0], 'r.')
                    else:
                        ax.plot(self.tp[:,1], self.tp[:,0], 'r.')
        plt.ion()
        plt.show()
        return fig, ax

    def play_video(self, video, frame_range, interval=30, points = None, axis = None, show_saturation = False, bit_depth = 16, color = 'r', include_W = False, roi_size = (11,11), save_fig_ax = False):
        def find_W(points_i):
            X = np.array([])
            Y = np.array([])
            for point in points_i:
                y, x = np.round(point).astype(int)
                h, w = np.array(roi_size).astype(int)
                y0 = y-h//2 - 0.5
                x0 = x-w//2 - 0.5
                y1 = y+h//2 + 0.5
                x1 = x+w//2 + 0.5
                X = np.hstack((X, np.array([x0, x1, x1, x0, x0, np.nan])))
                Y = np.hstack((Y, np.array([y0, y0, y1, y1, y0, np.nan])))
            return X, Y
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(video.mraw[frame_range[0]], cmap='gray')
        text = ax.text(0.65, 0.05, '', transform=ax.transAxes, color='black', ha='right', va='bottom')
        if axis is not None:
            ax.set_xlim(axis[0])
            ax.set_ylim(axis[1])
        if show_saturation:
            over_sat = np.where(video.mraw[frame_range[0]] > int(0.99*(2**bit_depth-1)))
            over_sat_plot = ax.plot(over_sat[1], over_sat[0], 'b.', alpha=0.2)
            under_sat = video.mraw[frame_range[0]] < int(0.01*(2**bit_depth-1))
            under_sat_plot = ax.plot(under_sat[1], under_sat[0], 'g.', alpha=0.2)
        if points is not None:
            pts = ax.plot(points[:,0,1], points[:,0,0], '.', color=color)
        if include_W:
            X, Y = find_W(points[:,0,:])
            W_plot, = ax.plot(X, Y, 'r-')

        def update(i):
            im.set_data(video.mraw[i])
            text.set_text(f'Frame {i}')
            if show_saturation:
                over_sat = np.where(video.mraw[i] > int(0.99*(2**bit_depth-1)))
                over_sat_plot[0].set_data(over_sat[1], over_sat[0])
                under_sat = np.where(video.mraw[i] < int(0.01*(2**bit_depth-1)))
                under_sat_plot[0].set_data(under_sat[1], under_sat[0])
            if include_W:
                X, Y = find_W(points[:,i,:])
                W_plot.set_data(X, Y)
            if points is not None:
                pts[0].set_data(points[:,i,1], points[:,i,0])
                if include_W:
                    return im, text, pts[0], W_plot
                return im, text, pts
            return im, text
    
        ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=interval)
        if save_fig_ax:
            ani.fig = fig
            ani.ax = ax
        plt.show()
        return ani

    def play_video_movement(self, video, frame_range, frame_delay, interval=30, points = None, axis = None,  bit_depth = 16, v_lims = None, include_W = False, roi_size = (9,9), include_G = False):              
        def find_W(points_i):
            X = np.array([])
            Y = np.array([])
            for point in points_i:
                y, x = np.round(point).astype(int)
                h, w = np.array(roi_size).astype(int)
                y0 = y-h//2 - 0.5
                x0 = x-w//2 - 0.5
                y1 = y+h//2 + 0.5
                x1 = x+w//2 + 0.5
                X = np.hstack((X, np.array([x0, x1, x1, x0, x0, np.nan])))
                Y = np.hstack((Y, np.array([y0, y0, y1, y1, y0, np.nan])))
            return X, Y
        
        def find_G(point,i):
            i, j = np.round(point).astype(int)
            h, w = np.array(roi_size).astype(int)
            i0 = i-h//2-1
            j0 = j-w//2-1
            i1 = i+h//2+1
            j1 = j+w//2+1
            frame = video.mraw[i][i0:i1,j0:j1]
            G = np.array([np.average(np.gradient(frame, axis=0)[1:-1]), np.average(np.gradient(frame, axis=1)[1:-1])])
            return G

        if v_lims is not None:
            vmin = v_lims[0]
            vmax = v_lims[1]
        else:
            vmin = 0
            vmax = 2**bit_depth-1
        fig, ax = plt.subplots()
        frame0 = ((2**bit_depth-1) - video.mraw[frame_range[0]])/2
        frame1 = (video.mraw[frame_range[frame_delay]])/2
        frame = frame1 + frame0
        im = ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, animated=True)
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, color='black', ha='right', va='bottom')
        if axis is not None:
            ax.set_xlim(axis[0])
            ax.set_ylim(axis[1])
        
        if points is not None:
            pts = ax.plot(points[:,0,1], points[:,0,0], 'r.')
        
        if include_W:
            X, Y = find_W(points[:,0,:])
            W_plot, = ax.plot(X, Y, 'r-')
        
        if include_G:
            Gi, Gj = find_G(points[0, 0], frame_range[0])
            G_plot, = ax.plot([points[0,0,1], points[0,0,1]+Gj], [points[0,0,0], points[0,0,0]+Gi], 'b-')

        def update(i):
            frame0 = ((2**bit_depth-1) - video.mraw[i])/2
            frame1 = (video.mraw[i+frame_delay])/2
            frame = frame1 + frame0
            im.set_data(frame)
            text.set_text(f'Frame {i}')
            if include_W:
                X, Y = find_W(points[:,i,:])
                W_plot.set_data(X, Y)
            if include_G:
                Gi, Gj = find_G(points[0,i], i)
                G_plot.set_data([points[0,i,1], points[0,i,1]+Gj], [points[0,i,0], points[0,i,0]+Gi])
            if points is not None:
                pts[0].set_data(points[:,i,1], points[:,i,0])
                if include_W:
                    if include_G:
                        return im, text, pts[0], W_plot, G_plot
                    return im, text, pts[0], W_plot
                return im, text, pts
            return im, text

        ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=interval)
        plt.show()
        return ani

    def play_video_local(self, video, frame_range, frame_delay, point, d, roi_size, interval=30, bit_depth = 16, v_lims = None):
        if v_lims is not None:
            vmin = v_lims[0]
            vmax = v_lims[1]
        else:
            vmin = 0
            vmax = 2**bit_depth-1
        fig, ax = plt.subplots(2,1)
        y, x = (point).astype(int)
        h, w = np.array(roi_size).astype(int)
        y0 = y-h//2
        x0 = x-w//2
        y1 = y+h//2
        x1 = x+w//2
        center = [h//2-0.5, w//2-0.5]
        # ax.set_xlim([x0, x1])
        # ax.set_ylim([y1, y0])
        current_frame = video.mraw[frame_range[0]][x0:x1,y0:y1]
        frame0 = ((2**bit_depth-1) - current_frame)/2
        frame1 = (video.mraw[frame_range[frame_delay]][x0:x1,y0:y1])/2
        frame = frame1 + frame0
        Gi      = np.average(np.gradient(frame, axis=0))
        Gj      = np.average(np.gradient(frame, axis=1))
        dd      = d[0] - d[1]
        quiverG = ax[1].plot([center[0], center[0]+Gj]      , [center[1], center[1]+Gi]      , 'b')
        quiverD = ax[0].plot([center[0], center[0]+dd[1]], [center[1], center[1]+dd[0]], 'r')

        im_i = ax[0].imshow(current_frame, cmap='gray', vmin=0, vmax=2**bit_depth-1, animated=True)
        im_m = ax[1].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, animated=True)
        
        def update(i):
            y, x = np.round(point+d[i]).astype(int)
            y0 = y-h//2
            x0 = x-w//2
            y1 = y+h//2
            x1 = x+w//2
            current_frame = video.mraw[i][x0:x1,y0:y1]
            frame0 = ((2**bit_depth-1) - current_frame)/2
            frame1 = (video.mraw[i+frame_delay][x0:x1,y0:y1])/2
            frame = frame1 + frame0
            im_i.set_data(current_frame)
            im_m.set_data(frame)
            Gi      = np.average(np.gradient(frame, axis=0))
            Gj      = np.average(np.gradient(frame, axis=1))
            dd      = d[i] - d[i+1]
            # quiverG.set_data([center[0], center[0]+Gj]      , [center[1], center[1]+Gi])
            # quiverD.set_data([center[0], center[0]+dd[i, 1]], [center[1], center[1]+dd[i, 0]])
            return im_i, im_m#, quiverG, quiverD

        ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=interval)
        plt.show()
        return ani

    def gradient_check_LK(self, ref_imgs, bit_depth=16, lim=60000, scale=0.3):
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
            self.eig_vals, self.eig_vecs = np.linalg.eigh(A.T @ A)
            print(np.linalg.inv(A.T @ A))
            print(self.eig_vals)
            ax[ax_i,2].plot(grad[1], grad[0],'k.')
            ax[ax_i,2].plot([0, -scale*self.eig_vals[0]**0.5*self.eig_vecs[0,0]], [0, -scale*self.eig_vals[0]**0.5*self.eig_vecs[0,1]],'r')
            ax[ax_i,2].plot([0, -scale*self.eig_vals[1]**0.5*self.eig_vecs[1,0]], [0, -scale*self.eig_vals[1]**0.5*self.eig_vecs[1,1]],'r')
            ax[ax_i,2].set_aspect('equal')
            ax[ax_i,2].set_xlim(-lim, lim)
            ax[ax_i,2].set_ylim(lim, -lim)
            ax[ax_i,2].set_xlabel('Ix')
            ax[ax_i,2].set_ylabel('Iy')
        plt.show()

    def nut_idx(self, point, exclude_high_amplitude = False, d_lim = 15):
        tp_temp = np.copy(self.tp)
        if exclude_high_amplitude:
            if self.d is None:
                self.open_displacements()
            self.exclude_tp(d_lim)
            tp_temp[~self.exclude_high_amplitude] = [0, 0]
        distances = cdist([np.flip(point)], tp_temp)
        self.nearest_nut_index = np.argmin(distances)

    def initialize_signals(self):
        """
        Prepares the force and accelerations signal for the impact data.
        Args:
            impact_data (dict): The impact data.
            approx_peak_heigth (float): The approximate peak height of the force signal.
        """
        self.fs_force   = self.impact_data['NI']['sample_rate']
        force_index     = self.impact_data['NI']['channel_names'].index('Force')
        self.force_raw      = self.impact_data['NI']['data'][:,force_index]
        self.t_force_raw    = self.impact_data['NI']['time']
        # start_index     = np.where(self.t_force_raw >= approx_peak_time)[0][0]
        # impact_mean     = np.mean(self.force_raw[start_index+2:start_index+1000])
        # self.force_raw      -= impact_mean

        # accel_index     = self.impact_data['NI']['channel_names'].index('Acceleration')
        # self.accel_raw      = self.impact_data['NI']['data'][:,accel_index]
        # mean_accel      = np.mean(self.accel_raw[start_index+2:start_index+1000])
        # self.accel_raw      = -(self.accel_raw-mean_accel)/9.81
        return
    
    def substract_mean(self, signal, start_index, end_index):
        """
        Substract the mean of the signal between the given indices.
        Args:
            signal (np.array): The signal to substract the mean from.
            start_index (int): The start index of the mean.
            end_index (int): The end index of the mean.
        """
        mean = np.mean(signal[start_index:end_index])
        return signal - mean

    def initialize_displacement(self, idx='all', dir='y'):
        """
        Prepares the displacement signal for the impact data.
        Args:
            displacements (np.array): The displacement signal.
            idx (int): The index of the displacement signal to use.
            dir (str): The direction of the displacement signal to use.
        """
        if self.d is None:
            warn.warn('No displacement data found. Run open_displacements first.')
            return
        if idx == 'all' and dir == 'y':
            self.displacements_raw =  self.d[:, :, 0]
            return
        elif idx == 'all' and dir == 'x':
            self.displacements_raw =  self.d[:, :, 1]
            return 
        elif idx == 'all' and dir == 'xy':
            self.displacements_raw =  self.d[:, :, :]
            return
        if dir == 'y':
            self.displacements_raw =  self.d[:, idx, 0, np.newaxis] if len(idx) > 0 else self.d[:, idx, 0]
        elif dir == 'x':
            self.displacements_raw =  self.d[:, idx, 1, np.newaxis] if len(idx) > 0 else self.d[:, idx, 1]
        elif dir == 'xy':
            self.displacements_raw =  self.d[:, idx, :, np.newaxis] if len(idx) > 0 else self.d[:, idx, :]

    def find_signal_start(self, signal, approximate_height = 0.5, approximate_distance = 8, treshold = 0.001, peak_n = 1):
        """
        Find the index where the signal starts.
        Args:
            signal (np.array): The signal to find the start of.
            approximate_height (float): The approximate peak height of the signal.
            approximate_distance (float): The approximate distance between peaks of the signal.
            treshold (float): The value concidered to be near zero to use for the signal.
        """
        peaks, F_peaks = find_peaks(signal, height=approximate_height, distance=approximate_distance)
        self.F_peak = F_peaks['peak_heights'][peak_n - 1]
        near_zero_indices = np.where(signal[:peaks[peak_n - 1]] < treshold)[0]
        first_zero_index = near_zero_indices[-1]
        return first_zero_index

    def greatest_common_divisor(self, fs_camera, fs_force):
        """
        Find the greatest common divisor of two numbers.
        Args:
            a (int): The first number.
            b (int): The second number.
        """
        self.frames_per_gcd     = int(fs_camera / math.gcd(int(fs_camera), int(fs_force)))
        return self.frames_per_gcd

    def n_samples_force_to_camera(self, n_samples_force):
        """
        Convert the number of samples of the force signal to the number of samples of the camera signal.
        Args:
            n_samples_force (int): The number of samples of the force signal.
        """
        return int(n_samples_force * self.fs_camera / self.fs_force)
    
    def n_samples_camera_to_force(self, n_samples_camera):
        """
        Convert the number of samples of the camera signal to the number of samples of the force signal.
        Args:
            n_samples_camera (int): The number of samples of the camera signal.
        """
        return int(n_samples_camera * self.fs_force / self.fs_camera)

    def shift_time(self, time, shift):
        """
        Shift the time signal by the given shift.
        Args:
            time (np.array): The time signal.
            shift (float): The shift to apply to the time signal.
        """
        return np.round(time - shift, 7)

    def clip_signal_before(self, signal, n_samples):
        """
        Clip the signal to the given number of samples.
        Args:
            signal (np.array): The signal to clip.
            n_samples (int): The number of samples to clip the signal to.
        """
        if len(signal.shape) == 1:
            return signal[n_samples:]
        else:
            return signal[:, n_samples:]

    def zero_signal_before(self, signal, n_samples):
        """
        Zero the signal before the given number of samples.
        Args:
            signal (np.array): The signal to zero.
            n_samples (int): The number of samples to zero the signal before.
        """
        if len(signal.shape) == 1:
            signal[:n_samples] = 0
        else:
            signal[:, :n_samples] = 0
        return signal
    
    def zero_signal_after(self, signal, n_samples):
        """
        Zero the signal after the given number of samples.
        Args:
            signal (np.array): The signal to zero.
            n_samples (int): The number of samples to zero the signal after.
        """
        if len(signal.shape) == 1:
            signal[n_samples:] = 0
        else:
            signal[:, n_samples:] = 0
        return signal

    def clip_signal_after(self, signal, first_zero_index):
        """
        Clip the signal after the given index.
        Args:
            signal (np.array): The signal to clip.
            first_zero_index (int): The index to clip the signal after.
        """
        if len(signal.shape) == 1:
            return signal[:first_zero_index + 1]
        else:
            return signal[:, :first_zero_index + 1]
    
    def zero_signal_treshold(self, signal, treshold):
        """
        Zero the signal where it is below the given treshold.
        Args:
            signal (np.array): The signal to zero.
            treshold (float): The treshold to use to zero the signal.
        """
        signal[signal < treshold] = 0
        return signal
    
    def find_last_common_time_ids(self, time_camera, time_force):
        """
        Find the last common time indices of the two time signals.
        Args:
            time_camera (np.array): The time signal of the camera.
            time_force (np.array): The time signal of the force sensor.
        """
        for i in range(1, self.frames_per_gcd+2): 
            if time_camera[-i] in time_force:
                index_force     = np.where(time_camera[-i] == time_force)[0][0]
                index_camera    = np.where(time_camera[-i] == time_camera)[0][0]
                
                if self.t_camera[index_camera] != self.t_force[index_force]:
                    warn.warn('Time signals do not end at the same time.')

                return index_force, index_camera
        return None
    
    def process_signals(self, first_zero_id_cam, first_zero_id_force):
        # max_d = np.max(np.abs(self.displacements_raw[self.nearest_nut_index,:,0]))
        # first_zero_id_cam = self.find_signal_start(self.displacements_raw[self.nearest_nut_index,:,0], treshold=0.08, approximate_height = max_d*.5, approximate_distance=100000)
        # first_zero_id_force = self.find_signal_start(self.force_raw, peak_n=peak_n, treshold=0.05, approximate_height = peak_F_threshold)
        # first_zero_id_force = self.find_signal_start(self.force_raw, peak_n=peak_n, treshold=0.05)
        zeros_camera    = self.greatest_common_divisor(self.fs_camera, self.fs_force)
        zeros_force     = self.n_samples_camera_to_force(zeros_camera)

        # Shift and align signals
        self.t_camera = self.shift_time(self.t_camera_raw, self.t_camera_raw[first_zero_id_cam-zeros_camera])
        self.t_force = self.shift_time(self.t_force_raw, self.t_force_raw[first_zero_id_force-zeros_force])

        # Clip signals
        self.t_camera          = self.clip_signal_before(self.t_camera,first_zero_id_cam-zeros_camera)
        # self.displacements     = self.clip_signal_before(self.displacements_raw, first_zero_id_cam-zeros_camera)
        self.displacements_x     = self.clip_signal_before(self.displacements_raw[:,:,1], first_zero_id_cam-zeros_camera)
        self.displacements_y     = self.clip_signal_before(self.displacements_raw[:,:,0], first_zero_id_cam-zeros_camera)
        self.t_force           = self.clip_signal_before(self.t_force, first_zero_id_force-zeros_force)
        self.force             = self.clip_signal_before(self.force_raw, first_zero_id_force-zeros_force)

        last_id_force, last_id_camera   = self.find_last_common_time_ids(self.t_camera, self.t_force)
        self.t_camera          = self.clip_signal_after(self.t_camera, last_id_camera)
        # self.displacements     = self.clip_signal_after(self.displacements, last_id_camera)
        self.displacements_x     = self.clip_signal_after(self.displacements_x, last_id_camera)
        self.displacements_y     = self.clip_signal_after(self.displacements_y, last_id_camera)
        self.t_force           = self.clip_signal_after(self.t_force, last_id_force)
        self.force             = self.clip_signal_after(self.force, last_id_force)

        # Zero Force after impact
        self.force = self.zero_signal_treshold(self.force, 0.1)

    def prepare_signals(self, peak_id = 0, threshold  = 0.001, approximate_force_peak_height = 0.5):
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
        # self.peak_id = peak_id
        self.threshold = threshold
        self.approximate_force_peak_height = approximate_force_peak_height
        self.frames_per_gcd     = int(self.fs_camera / math.gcd(int(self.fs_camera), int(self.fs_force)))
        n_zeros_before_cam      = self.frames_per_gcd
        n_zeros_before_for      = int(n_zeros_before_cam * self.fs_force / self.fs_camera)
        self.disp_peaks, _      = find_peaks(self.displacements_raw[self.nearest_nut_index], height=0.5, distance=self.fs_camera*0.01)
        near_zero_indices_cam   = np.where(self.displacements_raw[self.nearest_nut_index,:self.disp_peaks[0]] < threshold)[0]
        self.first_cam_near_zero_index = near_zero_indices_cam[-1]
        # Shift the time signal such that the impact - n_zeros_before_cam is at t=0, 
        self.t_camera           = np.round(self.t_camera_raw - self.t_camera_raw[self.first_cam_near_zero_index-n_zeros_before_cam], 7)
        # Clip the time
        self.t_camera           = self.t_camera[self.first_cam_near_zero_index-n_zeros_before_cam:]
        self.displacements      = self.displacements_raw[:, self.first_cam_near_zero_index-n_zeros_before_cam:]
        self.displacements[:n_zeros_before_cam] = 0

        self.force_peaks, _             = find_peaks(self.force_raw, height=approximate_force_peak_height)
        near_zero_indices_for           = np.where(self.force_raw[:self.force_peaks[0]] < threshold)[0]
        self.first_for_near_zero_index  = near_zero_indices_for[-1]
        near_zero_indices_for_after     = (np.where(self.force_raw[self.force_peaks[0]+1:self.force_peaks[0]+10] < threshold)[0]) + self.force_peaks[0]+1
        self.first_for_near_zero_index_after = near_zero_indices_for[0]
        self.t_force_raw            = np.round(self.t_force_raw - self.t_force_raw[self.first_for_near_zero_index - n_zeros_before_for],7)
        t_force_index_end       = np.argmin(np.abs(self.t_force_raw - self.t_camera[-1]))

        self.t_force            = self.t_force_raw[self.first_for_near_zero_index- n_zeros_before_for:t_force_index_end]
        self.force              = self.force_raw[self.first_for_near_zero_index- n_zeros_before_for:t_force_index_end]

        # Cut off signals such that they end at the same time. Loop is because sample ratios 
        for i in range(1, self.frames_per_gcd+2): 
            if self.t_camera[-i] in self.t_force:
                index = np.where(self.t_camera[-i] == self.t_force)[0][0]
                self.t_camera       = self.t_camera[:-i+1]
                # disp_nut_clipped = disp_nut_clipped[:-i+1]
                self.displacements  = self.displacements[:, :-i+1]
                self.t_force        = self.t_force[:index+1]
                self.force          = self.force[:index+1]
                break
        if self.t_camera[-1] != self.t_force[-1]:
            warn.warn('Time signals do not end at the same time. This is probably due to a rounding error. The last sample of the force signal is removed.')

        # Zero the force signal around the impact (The rest is all noise)
        self.force[self.force < 0.1] = 0
        return
    
    def set_freq_properties(self, padding_ratio = 1):
        self.n_d, self.freq_camera  = self._define_frequency_spectrum(self.t_camera, self.fs_camera, padding_ratio)
        # self.disp_fft               = self._FFT_on_signal(self.displacements, self.n_d)
        self.disp_fft_x               = self._FFT_on_signal(self.displacements_x, self.n_d)
        self.disp_fft_y               = self._FFT_on_signal(self.displacements_y, self.n_d)
        
        self.n_f, self.freq_force   = self._define_frequency_spectrum(self.t_force, self.fs_force, padding_ratio)
        self.force_fft              = self._FFT_on_signal(self.force, self.n_f)

        n_freq = len(self.freq_camera)
        self.freq_force = self.freq_force[:n_freq]
        # self.disp_fft[1:] *= 2
        self.disp_fft_x[1:] *= 2
        self.disp_fft_y[1:] *= 2
        self.force_fft[1:] *= 2

        # self.disp_fft = self.disp_fft[:,:n_freq]
        self.disp_fft_x = self.disp_fft_x[:,:n_freq]
        self.disp_fft_y = self.disp_fft_y[:,:n_freq]
        self.force_fft = self.force_fft[:n_freq]
        return
    
    def _define_frequency_spectrum(self ,signal, fs, padding_ratio = 1):
        """
        Defines the frequency spectrum for the given signal.
        Args:
            signal (np.array): The signal to define the frequency spectrum for.
            fs (float): The sampling frequency of the signal.
            extension_ratio (float): The ratio to extend the signal with.
        """
        n = len(signal)
        n = int(np.ceil(n*padding_ratio))
        return n, np.fft.rfftfreq(n, 1/fs)

    def _FFT_on_signal(self, signal, n_d = None):
        if n_d is None:
            n_d = len(signal)
        return np.fft.rfft(signal, n=n_d) / n_d

    def get_transfer_function(self, direction = 'y'):
        """
        Find transfer function between force and displacement
        Args:
            Disp (np.array): The displacement signal.
            Force (np.array): The force signal.
            Acquisition_period (float): The acquisition period of the signals.
            type (str): The type of transfer function to calculate.
        """
        Acquisition_period  = self.t_camera[-1]
        if direction == 'y':
            S_xx = 1/Acquisition_period * np.conj(self.disp_fft_y) * self.disp_fft_y
            S_ff = 1/Acquisition_period * np.conj(self.force_fft) * self.force_fft
            S_xf = 1/Acquisition_period * np.conj(self.disp_fft_y) * self.force_fft
            S_fx = 1/Acquisition_period * np.conj(self.force_fft) * self.disp_fft_y
        elif direction == 'x':
            S_xx = 1/Acquisition_period * np.conj(self.disp_fft_x) * self.disp_fft_x
            S_ff = 1/Acquisition_period * np.conj(self.force_fft) * self.force_fft
            S_xf = 1/Acquisition_period * np.conj(self.disp_fft_x) * self.force_fft
            S_fx = 1/Acquisition_period * np.conj(self.force_fft) * self.disp_fft_x
        self.H1 = S_fx / S_ff
        self.H2 =  S_xx / S_xf
        self.Hv =  (S_fx / S_ff * S_xx / S_xf)**0.5
        return
    
    def H_outliers(self, z_limit = 1., h_type = 'H1', f_range=(3, 50)):
        """
        Removes outliers from the transfer function.
        Args:
            H (np.array): The transfer function.
            z_limit (float): The threshold to use to remove outliers.
        """
        f_start = self.freq_camera.searchsorted(f_range[0])
        f_end = self.freq_camera.searchsorted(f_range[1])
        if h_type == 'H1':
            z_scores = stats.zscore(np.abs(self.H1[:, f_start:f_end]), axis=0)
        elif h_type == 'H2':
            z_scores = stats.zscore(self.H2, axis=0)
        elif h_type == 'Hv':
            z_scores = stats.zscore(self.Hv, axis=0)
        z_scores = np.mean(np.abs(z_scores), axis=1)
        self.exclude_outliers = z_scores < z_limit
        return

    def exclude_tp(self, d_lim = 15):
        self.d_lim = d_lim
        self.exclude_high_amplitude = np.max(np.linalg.norm(self.d, axis=2),1)<d_lim
        return
    
    def valid_tp(self, d_lim = 15, z_limit = 1., h_type = 'H1', d_min = None, f_range=(3, 50)):
        self.exclude_tp(d_lim)
        self.H_outliers(z_limit, h_type, f_range)
        self.valid_tps = self.exclude_high_amplitude & self.exclude_outliers
        if d_min is not None:
            self.valid_tps = self.valid_tps & (np.max(np.linalg.norm(self.d, axis=2),1) > d_min)
        return

    def save(self, root = None, file_name = None):
        if root is not None:
            root_EMA_struct = root
        else:
            root_EMA_struct = self.root_EMA_struct

        if file_name is not None:
            file_name = self.file_name
    
        with open(os.path.join(root_EMA_struct, file_name + '_EMA_structure.pkl'), 'wb') as f:
            pkl.dump(self, f)
        return
    
    def plot_hub_modes(self, cam, fig, ax_real, ax_imag, mode_number, video, D_in = 70, D_out = 100, nut_wh = (10, 60), account_for_distortion = True, normalize_colors = True):
        still_frame = video.reader.get_frame(0)
        if account_for_distortion:
            file_path = os.path.splitext(video.cih_file)[0] + '_dist.pkl'
            with open(file_path, 'rb') as f:
                data = pkl.load(f)
            homography_matrix = data['homography_matrix']
            still_frame = cv2.warpPerspective(still_frame, homography_matrix, (still_frame.shape[1], still_frame.shape[0]))
            spider_ij = np.array([self.spider_ij[0], self.spider_ij[1], 1])
            spider_ij = np.dot(homography_matrix, spider_ij)
            spider_ij = np.array([spider_ij[0]/spider_ij[2], spider_ij[1]/spider_ij[2]])
            prey_ij = np.array([self.prey_ij[0], self.prey_ij[1], 1])
            prey_ij = np.dot(homography_matrix, prey_ij)
            prey_ij = np.array([prey_ij[0]/prey_ij[2], prey_ij[1]/prey_ij[2]])
            tp = np.dot(homography_matrix, np.array([self.tp[:, 1], self.tp[:, 0], np.ones(self.tp.shape[0])]))
            tp = (tp[1::-1]/tp[2][np.newaxis, :]).T
            
            self.prey_ij_dist = prey_ij
            self.spider_ij_dist = spider_ij
            self.tp_dist = tp
            prey_ij = self.prey_ij_d
            spider_ij = self.spider_ij_d
        else:
            spider_ij = self.spider_ij
            prey_ij = self.prey_ij
            tp = self.tp

        _, _, _, full_mask = self.hub_classifier( D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = account_for_distortion)

        cmap_circ = cm.get_cmap('Accent')
        norm = col.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm = norm, cmap=cmap_circ)

        ax_real.imshow(still_frame, cmap='gray')
        ax_imag.imshow(still_frame, cmap='gray')
        fig, ax_real, ax_imag = plot_mode_shape_flat(cam, fig, ax_real, ax_imag, mode_number, tp[self.valid_tps], full_mask[self.valid_tps], normalize_colors = normalize_colors)
        ax_real.set_aspect('equal')
        ax_real.set_xlim(spider_ij[0]-D_out*0.75, spider_ij[0]+D_out*0.75)
        ax_real.set_ylim(spider_ij[1]+D_out*0.75, spider_ij[1]-D_out*0.75)
        ax_real.plot([spider_ij[0], prey_ij[0]], [spider_ij[1], prey_ij[1]], 'r-', lw = 5)
        ax_imag.set_aspect('equal')
        ax_imag.set_xlim(spider_ij[0]-D_out*0.75, spider_ij[0]+D_out*0.75)
        ax_imag.set_ylim(spider_ij[1]+D_out*0.75, spider_ij[1]-D_out*0.75)
        ax_imag.plot([spider_ij[0], prey_ij[0]], [spider_ij[1], prey_ij[1]], 'r-', lw = 5)

    def plot_hub_features(self, fig, ax, video, D_in = 70, D_out = 100, nut_wh = (10, 60), n_sections = 8, account_for_distortion = True):
        """ 
        Plot the hub features.
        Args:
            video (Video): The video object. Function get first frame of the video.
            height_pixels (int): The height of the elipse in pixels.
            width_elipse (int): The width of the elipse in pixels.
            n_sections (int): The number of sections to divide the elipse in.    
        """
        still_frame = video.reader.get_frame(0)
        if account_for_distortion:
            file_path = os.path.splitext(video.cih_file)[0] + '_dist.pkl'
            with open(file_path, 'rb') as f:
                data = pkl.load(f)
            homography_matrix = data['homography_matrix']
            still_frame = cv2.warpPerspective(still_frame, homography_matrix, (still_frame.shape[1], still_frame.shape[0]))
            spider_ij = np.array([self.spider_ij[0], self.spider_ij[1], 1])
            spider_ij = np.dot(homography_matrix, spider_ij)
            spider_ij = np.array([spider_ij[0]/spider_ij[2], spider_ij[1]/spider_ij[2]])
            prey_ij = np.array([self.prey_ij[0], self.prey_ij[1], 1])
            prey_ij = np.dot(homography_matrix, prey_ij)
            prey_ij = np.array([prey_ij[0]/prey_ij[2], prey_ij[1]/prey_ij[2]])
            tp = np.dot(homography_matrix, np.array([self.tp[:, 1], self.tp[:, 0], np.ones(self.tp.shape[0])]))
            tp = (tp[1::-1]/tp[2][np.newaxis, :]).T
            
            self.prey_ij_dist = prey_ij
            self.spider_ij_dist = spider_ij
            self.tp_dist = tp
            prey_ij = self.prey_ij_d
            spider_ij = self.spider_ij_d
        else:
            spider_ij = self.spider_ij
            prey_ij = self.prey_ij
            tp = self.tp

        inside_and_valid, angle_sections, bucket_edges = self.section_classifier(n_sections = n_sections, D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = account_for_distortion)
        self.inside_and_valid = inside_and_valid
        self.angle_sections = angle_sections
        self.d_hub_x = self.displacements_x[inside_and_valid]
        self.d_hub_y = self.displacements_y[inside_and_valid]
        cmap_circ = cm.get_cmap('Accent')
        norm = col.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm = norm, cmap=cmap_circ)
        self.n_sections = n_sections
        # fig, ax = plt.subplots(figsize = (10,10))
        ax.imshow(still_frame, cmap='gray')
        ax.set_aspect('equal')
        ax.set_xlim(spider_ij[0]-D_out*0.75, spider_ij[0]+D_out*0.75)
        ax.set_ylim(spider_ij[1]+D_out*0.75, spider_ij[1]-D_out*0.75)
        ax.plot([spider_ij[0], prey_ij[0]], [spider_ij[1], prey_ij[1]], 'r-', lw = 5)
        ax.scatter(tp[self.valid_tps,1], tp[self.valid_tps,0], s = 2, c = 'black')
        for section in range(n_sections):
            current_points = angle_sections == section
            ax.scatter(tp[inside_and_valid,1][current_points], tp[inside_and_valid,0][current_points], s = 14, c = 'white')
            ax.scatter(tp[inside_and_valid,1][current_points], tp[inside_and_valid,0][current_points], s = 10, c = sm.to_rgba(section/(n_sections-1)))
        circle = patches.Circle((spider_ij[0], spider_ij[1]), D_out/2, edgecolor='blue', facecolor='none', lw=1)
        ax.add_patch(circle)
        circle = patches.Circle((spider_ij[0], spider_ij[1]), D_in/2, edgecolor='blue', facecolor='none', lw=1)
        ax.add_patch(circle)

        circle_center = (spider_ij[0], spider_ij[1])
        rect_x_min = circle_center[0] - nut_wh[0] / 2
        rect_x_max = circle_center[0] + nut_wh[0] / 2
        rect_y_min = circle_center[1] - 3*nut_wh[1] / 4
        rect_y_max = circle_center[1] + nut_wh[1] / 4
        ax.plot([rect_x_min, rect_x_max, rect_x_max, rect_x_min, rect_x_min], [rect_y_min, rect_y_min, rect_y_max, rect_y_max, rect_y_min], 'b-', lw=1)
        
        # TEMP
        # ax.plot(self.prey_ij_dist[0], self.prey_ij_dist[1], 'ro')
        # ax.plot(self.spider_ij_dist[0], self.spider_ij_dist[1], 'bo')
        # ax.scatter(self.tp_dist[:,1], self.tp_dist[:,0], s = 14, c = 'black')
        return fig, ax
    
    def plot_hub_disp(self,fig, ax, t_max = None, d_max = 'auto', normalized = True, style = 'translation', legend = True, title = True, xlabel = True, lim_lines = False, plot_nut_disp = False):
        """
        Plot the displacement of the hub."""
        cmap_circ = cm.get_cmap('Accent')
        norm = col.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm = norm, cmap=cmap_circ)
        if normalized:
            reaction_time = self.reaction_time/self.fn0*self.FN0
            time_vector = self.t_camera/self.fn0*self.FN0
            arg_rt_max = np.argmax(time_vector > reaction_time)
            d_norm = np.max(np.abs(self.d_hub_y)[:,:arg_rt_max])
            self.d_hub_y = self.d_hub_y/d_norm
            if xlabel:
                ax.set_xlabel(r'Time (scaled: $t\cdot 42.1 Hz/fn_0$)')
            ax.set_ylabel('scaled displacement (d/max(d))')
        else:
            reaction_time = self.reaction_time
            time_vector = self.t_camera
            arg_rt_max = np.argmax(time_vector > reaction_time)
            if xlabel:
                ax.set_xlabel('Time [s]')
            ax.set_ylabel('Displacement [pixels]')
            if t_max is not None:
                t_max = t_max/self.FN0*self.fn0
        
        if (style == 'rotation'or style == 'pitch_roll') and normalized:
            ax.set_ylabel('$\Delta$d normalized [-]')
        elif (style == 'rotation'or style == 'pitch_roll') and not normalized:
            ax.set_ylabel('$\Delta$d [pixels]')

        if t_max is None:
            t_max = time_vector[-1]

        arg_t_max = np.argmax(time_vector > t_max)
        if d_max == 'auto':
            d_max = np.max(np.abs(self.d_hub_y)[:,:arg_t_max])
        elif d_max == 'd_lim':
            d_max = self.d_lim

        # fig, ax = plt.subplots(figsize = (16, 8))
        t_max_vec = []
        d_max_vec = []
        if style == 'translation':
            for section in range(self.n_sections):
                current_points = self.angle_sections == section
                if np.sum(current_points) == 0:
                    continue
                max_row_disp, max_col_disp = np.unravel_index(np.argmax(self.d_hub_y[current_points,:arg_rt_max].flatten()), self.d_hub_y[current_points,:arg_rt_max].shape)
                for i in range(self.d_hub_y[current_points].shape[0]):
                    ax.plot(time_vector, -self.d_hub_y[current_points][i],'--', color=sm.to_rgba(section/(self.n_sections-1)), alpha=0.5, lw=0.5)
                ax.plot(time_vector, -np.mean(self.d_hub_y[current_points], axis=0), color = sm.to_rgba(section/(self.n_sections-1)), lw = 3, label = f'mean section {section+1}', zorder=10)
                ax.scatter(time_vector[max_col_disp], -self.d_hub_y[current_points][max_row_disp, max_col_disp], marker='*', s = 80, color=sm.to_rgba(section/(self.n_sections-1)), label = f'Max. d in sec: {section+1}', zorder=12)
                t_max_vec.append(time_vector[max_col_disp])
                d_max_vec.append(-self.d_hub_y[current_points][max_row_disp, max_col_disp])
            if lim_lines:
                t0 = np.min(t_max_vec)
                t1 = np.max(t_max_vec)
                d0 = np.min(d_max_vec)
                d1 = np.max(d_max_vec)
                ax.vlines(t0, -d_max*1.05, d1, color='r', linestyle='--', zorder=11, label='min/max first peak')
                ax.vlines(t1, -d_max*1.05, d1, color='r', linestyle='--', zorder=11)
                ax.hlines(d0, 0, t1, color='r', linestyle='--', zorder=11)
                ax.hlines(d1, 0, t1, color='r', linestyle='--', zorder=11)
            
            if plot_nut_disp:
                ax.plot(time_vector, -self.displacements_y[self.nearest_nut_index]/np.max(np.abs(self.displacements_y)[self.nearest_nut_index,:arg_rt_max]), 'k:', lw=1, alpha = 0.7, label='Nut displacement', zorder=9)

        elif style == 'rotation':
            for section in range(self.n_sections//2):
                pointset1 = self.angle_sections == section
                pointset2 = self.angle_sections == section + self.n_sections//2
                mean_disp1 = np.mean(self.d_hub_y[pointset1], axis=0)
                mean_disp2 = np.mean(self.d_hub_y[pointset2], axis=0)
                ax.plot(time_vector, -mean_disp1 + mean_disp2, color = sm.to_rgba(section/(self.n_sections-1)), lw = 3, label = f'mean (sec {section+1} - sec {section+1+self.n_sections//2})', zorder=10)
        
        elif style == 'pitch_roll':
            pointset1 = self.angle_sections == 0
            pointset2 = self.angle_sections == self.n_sections//2
            mean_disp1 = np.mean(self.d_hub_y[pointset1], axis=0)
            mean_disp2 = np.mean(self.d_hub_y[pointset2], axis=0)
            ax.plot(time_vector, -mean_disp1 + mean_disp2, color = sm.to_rgba(0), lw = 3, label = f'mean pitch', zorder=10)

            pointset1 = self.angle_sections == self.n_sections//4
            pointset2 = self.angle_sections == 3*self.n_sections//4
            mean_disp1 = np.mean(self.d_hub_y[pointset1], axis=0)
            mean_disp2 = np.mean(self.d_hub_y[pointset2], axis=0)
            ax.plot(time_vector, -mean_disp1 + mean_disp2, color = sm.to_rgba(self.n_sections//4), lw = 3, label = f'mean roll', zorder=10)


        ax.vlines(reaction_time, -d_max*1.05, d_max*1.05, color='k', linestyle='--', label='Reaction time (scaled) (100ms)')
         
        if title:
            if type(title) == str:
                ax.set_title(title)
            else:
                ax.set_title(self.file_name)
        ax.set_xlim([0, t_max])
        ax.set_ylim([-d_max*1.05, d_max*1.05])
        if legend:
            legend = ax.legend(loc = 'lower center', ncols = self.n_sections//2 + 1, bbox_to_anchor=(0.5, 0.9))
            legend.set_zorder(100)

        return fig, ax, np.array(t_max_vec), np.array(d_max_vec)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return
    
    def hub_classifier(self, D_in = 50, D_out = 100, nut_wh = (10, 60), account_for_distortion = True):
        self.D_in = D_in
        self.D_out = D_out
        self.nut_wh = nut_wh
        if account_for_distortion:
            spider_ij = self.spider_ij_d
            tp = self.tp_dist
        else:
            spider_ij = self.spider_ij
            tp = self.tp
        circle_center = (spider_ij[0], spider_ij[1])
        a_in = D_in / 2
        a_out = D_out / 2

        # Rectangle bounds based on the circle center
        rect_x_min = circle_center[0] - nut_wh[0] / 2
        rect_x_max = circle_center[0] + nut_wh[0] / 2
        rect_y_min = circle_center[1] - 3*nut_wh[1] / 4
        rect_y_max = circle_center[1] + nut_wh[1] / 4

        # Exclude points inside the rectangle
        outside_rectangle = np.logical_or.reduce((
                            tp[:, 1] < rect_x_min,
                            tp[:, 1] > rect_x_max,
                            tp[:, 0] < rect_y_min,
                            tp[:, 0] > rect_y_max
                            ))
        inside_circle = (tp[:,1] - circle_center[0])**2 / a_in**2 + (tp[:,0] - circle_center[1])**2 / a_in**2 >= 1
        outside_circle = (tp[:,1] - circle_center[0])**2 / a_out**2 + (tp[:,0] - circle_center[1])**2 / a_out**2 <= 1
        full_mask = inside_circle & outside_circle & outside_rectangle
        inside_and_valid = self.valid_tps & full_mask
        return tp, inside_and_valid, spider_ij, full_mask

    def section_classifier(self, n_sections = 8, D_in = 50, D_out = 100, nut_wh = (10, 60), account_for_distortion = True):
        self.D_in = D_in
        self.D_out = D_out
        self.nut_wh = nut_wh
        if account_for_distortion:
            prey_ij = self.prey_ij_d
        else:
            prey_ij = self.prey_ij
        tp, inside_and_valid, spider_ij, _ = self.hub_classifier( D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = account_for_distortion)
        points_inside_ellipse = tp[inside_and_valid]
        angles = np.arctan2(-(points_inside_ellipse[:, 0] - spider_ij[1]), points_inside_ellipse[:, 1] - spider_ij[0])

        angle_prey = np.arctan2(-(prey_ij[1] - spider_ij[1]), prey_ij[0] - spider_ij[0])
        if angle_prey < 0:
            angle_prey += 2*np.pi
        angle_prey = angle_prey/(2*np.pi)
        angles[angles < 0] = angles[angles < 0] + 2*np.pi
        normalized_angles = angles/(2*np.pi)

        bucket_edges = np.linspace(-1/(2*n_sections), 1-1/(2*n_sections), num=n_sections + 1) + angle_prey
        bucket_edges = bucket_edges - np.floor(bucket_edges)
        bucket_edges[bucket_edges > 1] -= 1
        bucket_edges[0] = 0
        bucket_edges = np.append(bucket_edges,1)
        bucket_edges = np.sort(bucket_edges)
        angle_sections = (np.digitize(normalized_angles, bucket_edges, right = True)-1).astype(int)
        angle_sections[angle_sections == n_sections] = 0
        start_index = np.searchsorted(bucket_edges, angle_prey, side='right') - 1
        angle_sections = np.mod(angle_sections - start_index, n_sections)
        bucket_edges = np.roll(bucket_edges[1:-1], -start_index)
        return inside_and_valid, angle_sections, bucket_edges

    def plot_Sxx_freq(self, ax_Sxx, S_xx, n_sections = 8, D_in = 50, D_out = 100, nut_wh = (10, 60), account_for_distortion = True):
        cmap_circ = cm.get_cmap('Accent')
        norm = col.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm = norm, cmap=cmap_circ)
        inside_and_valid, angle_sections, _ = self.section_classifier(n_sections = n_sections, D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = account_for_distortion)
        for section in range(n_sections):
            current_points = angle_sections == section
            if np.sum(current_points) == 0:
                continue
            S_xx_current = np.mean(np.abs(S_xx[inside_and_valid][current_points]), axis=0)
            ax_Sxx.plot(self.freq_camera, S_xx_current, c = sm.to_rgba(section/(n_sections-1)), label = f'Section {section+1}')

    def plot_Sxx_sections(self, fig, ax_imag, ax_real, S_xx_fn, n_sections = 8, D_in = 50, D_out = 100, nut_wh = (10, 60), account_for_distortion = True):
        overlay_ax_imag = fig.add_axes(ax_imag.get_position(), frame_on=False)
        overlay_ax_real = fig.add_axes(ax_real.get_position(), frame_on=False)
        cmap_circ = cm.get_cmap('Accent')
        inside_and_valid, angle_sections, bucket_edges = self.section_classifier(n_sections = n_sections, D_in = D_in, D_out = D_out, nut_wh = nut_wh, account_for_distortion = account_for_distortion)
        R = 0.9
        x = R * np.cos(bucket_edges * 2 * np.pi - np.pi/n_sections)
        y = R * np.sin(bucket_edges * 2 * np.pi - np.pi/n_sections)
        S_real = []
        # S_imag = []
        norm = col.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm = norm, cmap=cmap_circ)
        for section in range(n_sections):
            current_points = angle_sections == section
            if np.sum(current_points) == 0:
                S_real.append(np.nan)
                continue
            S_xx_current = S_xx_fn[inside_and_valid][current_points]
            S_real.append(np.mean(np.abs(S_xx_current), axis=0))
            # S_imag.append(np.mean(np.imag(S_xx_current), axis=0))

            overlay_ax_imag.plot([0, x[section]], [0, y[section]], c = sm.to_rgba(section/(n_sections-1)), lw = 1.5)
            overlay_ax_real.plot([0, x[section]], [0, y[section]], c = sm.to_rgba(section/(n_sections-1)), lw = 1.5)
        cmap_S = cm.get_cmap('plasma')
        # all_values = np.concatenate([S_real, S_imag])
        norm_S = col.Normalize(vmin=np.nanmin(S_real), vmax=np.nanmax(S_real))

        # Set the limits for circular overlay
        overlay_ax_imag.set_xlim(-1, 1)
        overlay_ax_imag.set_ylim(-1, 1)
        overlay_ax_imag.axis('off')

        overlay_ax_real.set_xlim(-1, 1)
        overlay_ax_real.set_ylim(-1, 1)
        overlay_ax_real.axis('off')

        # Scatter plots with normalized colors for S_imag and S_real on overlay axes
        # overlay_ax_imag.scatter(x, y, c=S_imag, cmap=cmap_S, norm=norm_S)
        overlay_ax_real.scatter(x, y, c=S_real, cmap=cmap_S, norm=norm_S)
            
    @staticmethod
    def load(file_name, root = None):
        if root is None:
            # root = r'G:/.shortcut-targets-by-id/1k1B8zPb3T8H7y6x0irFZnzzmfQPHMRPx/Illimited Lab Projects/Research Projects/Spiders/Simulations/EMA structure'
            root = r"C:/Users/thijsmas/Documents/GitHub/pyidi_data/EMA structure"#
            
        with open(os.path.join(root, file_name + '_EMA_structure.pkl'), 'rb') as f:
            return pkl.load(f)
        
def plot_FRF(cam, fig = None, ax = None, c = 'r', ls = '-', annotate = True, label = None):
    H, A = cam.get_constants(whose_poles='own', FRF_ind='all', least_squares_type='new')
    if fig is None:
        fig, ax = plt.subplots(2,1,figsize=(15, 9))
        ax[0].set_ylabel('H')
        ax[1].set_ylabel('H')
        ax[1].set_xlabel('Frequency [Hz]')
    y_cor_anot_lin = [3500, 1000, 2500, 1000, 1500, 500, 1000,1500,500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000,1500,500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000, 1500, 500, 1000]
    ax[0].plot(cam.freq, np.average(np.abs(H), axis=0), color=c, linestyle=ls, label=label)
    if annotate:
        for i, freq in enumerate(cam.nat_freq):
            ax[0].axvline(x=freq, color=c, linestyle=ls)
            ax[0].annotate(f"Mode {i+1}\n{freq:.2f} Hz", xy=(freq, y_cor_anot_lin[i%len(y_cor_anot_lin)]), xytext=(freq, y_cor_anot_lin[i%len(y_cor_anot_lin)]), ha='center', fontsize=6)
    ax[1].semilogy(cam.freq, np.average(np.abs(H), axis=0), color=c, linestyle=ls, label=label)
    if annotate:
        for i, freq in enumerate(cam.nat_freq):
            ax[1].axvline(x=freq, color=c, linestyle=ls)
    ax[0].legend()
    ax[1].legend()
    plt.show()
    return fig, ax

def plot_H(cam, fig = None, ax = None, c = 'r', ls = '-', label = None, annotate = False):
    H = np.average(cam.H, axis=0)
    if fig is None:
        fig, ax = plt.subplots(figsize=(15, 9))
    if annotate:
        for freq in cam.nat_freq:
            ax.axvline(x=freq, color=c, linestyle=':', linewidth=0.5)
    ax.set_ylabel('|H|')
    ax.set_xlabel('Frequency [Hz]')
    ax.semilogy(cam.freq, np.abs(H), color=c, linestyle=ls, label=label)
    return fig, ax

def plot_mode_shape_flat(cam, fig, ax_real, ax_imag, mode_number, tp, mask = None, normalize_colors = False):
    if mask is None:
        mask = np.ones(tp.shape[0], dtype=bool)
    tp_lim = tp[mask]
    A       = cam.A[mask, mode_number]
    A_real =  np.real(A)
    A_imag =  np.imag(A)
    cmap = plt.cm.get_cmap('plasma')
    if normalize_colors:
        min_A = np.min(np.abs(A))
        max_A = np.max(np.abs(A))
        A_real = (A_real - min_A) / (max_A - min_A)
        A_imag = (A_imag - min_A) / (max_A - min_A)
        ax_imag.set_title(f'min: {min_A:.2g}, max: {max_A:.2g}', fontsize=8)
    # norm = col.Normalize(vmin=np.min(np.abs(cam.A[:, mode_number])), vmax=np.max(np.abs(cam.A[:, mode_number])))
    ax_real.set_ylabel(f'Real: {mode_number+1} - {cam.nat_freq[mode_number]:.2f} Hz\n part: {np.abs(cam.partfactors[cam.pole_ind[mode_number][0]][cam.pole_ind[mode_number][1]]):.2f}')
    ax_real.scatter(tp_lim[:, 1], tp_lim[:, 0], c = A_real, cmap = cmap)#, norm=norm)
    # ax_real.axis('off')   
    ax_imag.set_ylabel(f'Imag: {mode_number+1} - {cam.nat_freq[mode_number]:.2f} Hz\n part: {np.abs(cam.partfactors[cam.pole_ind[mode_number][0]][cam.pole_ind[mode_number][1]]):.2f}')
    ax_imag.scatter(tp_lim[:, 1], tp_lim[:, 0], c = A_imag, cmap = cmap)#, norm=norm)
    # ax_imag.axis('off')
    return fig, ax_real, ax_imag

def plot_mode_shape(cam, mode_number, tp_lim, node, view=(28, -76), find_Z=False):
    A = cam.A
    A_imag = np.imag(A[:, mode_number]) / np.linalg.norm(A[:, mode_number])
    A_real = np.real(A[:, mode_number]) / np.linalg.norm(A[:, mode_number])

    An_i = np.linalg.norm(A[node, mode_number])
    An_r = np.linalg.norm(A[node, mode_number])
    t_vec = np.linspace(0, 2 * np.pi, 100)
    A_max = 0
    t_max = 0
    for t in t_vec:
        A = An_i * np.sin(t) + An_r * np.cos(t)
        if A < A_max:
            A_max = A
            t_max = t

    Z = A_imag * np.sin(t_max) + A_real * np.cos(t_max)
    if find_Z:
        return Z
    
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': '3d'})
    ax.set_title(f'Mode {mode_number+1} - {cam.nat_freq[mode_number]:.2f} Hz')
    ax.set_zlabel('Normalized mode shape')
    # Plot the initial position of the points
    ax.plot(tp_lim[:, 1], -tp_lim[:, 0], np.zeros_like(tp_lim[:, 0]), 'g.', markersize=0.5, alpha=0.5)
    ax.view_init(azim=view[1], elev=view[0])
    
    # Create a colormap ranging from blue to red
    cmap = plt.cm.get_cmap('seismic')

    mode_plot = ax.scatter(tp_lim[:, 1], -tp_lim[:, 0], Z, c=Z, cmap=cmap)
    plt.show()
    return fig, ax

def plot_MAC(cam, fig, ax, n_modes = 15, mode_vec = None, cmap='plasma'):
    MAC = cam.autoMAC()
    if mode_vec is not None:
        grid_x, grid_y = np.meshgrid(mode_vec, mode_vec)
        # MAC_copy = np.copy(MAC)
        MAC = MAC[grid_x, grid_y]
        n_modes = len(mode_vec)
    
    # Add row and column of zeros
    MAC = np.pad(MAC, ((1, 0), (1, 0)), mode='constant')
    
    # fig, ax = plt.subplots( figsize=(8, 8))
    im = ax.imshow(MAC, cmap=cmap)
    fig.colorbar(im, ax=ax)
    n_modes = min(n_modes, len(cam.nat_freq))
    ax.set_xlim([.5, n_modes+.5])
    ax.set_ylim([n_modes+.5, .5])
    if mode_vec is not None:
        ax.set_xticks(range(1, n_modes+1))
        ax.set_yticks(range(1, n_modes+1))
        mode_vec = [f'{mode+1}' for mode in mode_vec]
        ax.set_xticklabels(mode_vec)
        ax.set_yticklabels(mode_vec)
    return fig, ax
# def FRF_from_poles(cam):
#     fn     = cam.nat_fre
#     Xi     = cam.nat_xi
#     c      = [Xii / np.sqrt(1 - Xii**2) for Xii in Xi]

def find_green(cam, n_modes = None, plot=False):
    phi_X = cam.A
    phi_A = cam.A
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]
    
    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]
    
    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(f'Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})')

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(f'Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})')

    GREEN = np.zeros((phi_X.shape[1], phi_A.shape[1]), dtype=np.complex128)
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            GREEN[i, j] =     (np.conj(phi_X[:, i]) @ phi_X[:, i] *\
                            np.conj(phi_A[:, j]) @ phi_A[:, j])

    Green = np.pad(GREEN, ((1, 0), (1, 0)), mode='constant')
    
    if GREEN.shape == (1, 1):
        GREEN = GREEN[0, 0]

    if plot:
        if n_modes is None:
            n_modes = len(cam.nat_freq)
        fig, ax = plt.subplots( figsize=(8, 8))
        im = ax.imshow(np.real(Green), cmap='plasma')
        fig.colorbar(im, ax=ax)
        ax.set_xlim([.5, n_modes+.5])
        ax.set_ylim([n_modes+.5, .5])
        plt.show()
        return fig, ax, GREEN
    return GREEN

def animate_mode_shape(cam, mode_number, tp_lim, multiplier=1, indices_to_plot = None, frames = range(200), interval = 30, view = (28, -76)):
    if indices_to_plot is not None:
        A = cam.A[indices_to_plot, :]
        tp_lim = tp_lim[indices_to_plot]
    else:
        A = cam.A
    A_imag = np.imag(A[:,mode_number])/np.linalg.norm(A[:,mode_number])
    A_real = np.real(A[:,mode_number])/np.linalg.norm(A[:,mode_number])
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': '3d'})
    ax.set_title(f'Mode {mode_number+1} - {cam.nat_freq[mode_number]:.2f} Hz')
    ax.set_zlabel('Normalized mode shape')
    # Plot the inital position of the points
    ax.plot(tp_lim[:,1], -tp_lim[:,0], np.zeros_like(tp_lim[:,0]), 'g.', markersize=0.5, alpha=0.5)
    mode_plot, = ax.plot(tp_lim[:,1], -tp_lim[:,0], np.zeros_like(tp_lim[:,0]), 'r.', markersize=1)
    ax.view_init(azim=view[1], elev=view[0])
    def update(frame):
        Z = multiplier*(A_imag * np.sin(frame * np.pi / 100) + A_real * np.cos(frame * np.pi / 100))
        mode_plot.set_data(tp_lim[:,1], -tp_lim[:,0])
        mode_plot.set_3d_properties(Z)
        return mode_plot

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.show()
    return ani

def find_valid_points(cam, max_amplitude=5, exlude_outliers=True, z_limit=1.):
    valid_ids = np.all(np.abs(cam.A) <= max_amplitude, axis=1)
    not_nan_ids = ~np.isnan(cam.A, axis=1)
    valid_ids = valid_ids & not_nan_ids
    if exlude_outliers:
        valid_ids = valid_ids & ~H_outliers(cam.H, z_limit)
    return valid_ids

def get_PI_histogram(still_image, bins=100, remove_sides_fraction = None,  show=True, ax=None, return_ax=False, save_path = None, save_name = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if remove_sides_fraction is not None:
        remove_sides = int(remove_sides_fraction * still_image.shape[0])
        still_image = still_image[remove_sides:-remove_sides, remove_sides:-remove_sides]
    hist, bin_edges = np.histogram(still_image, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, hist)
    plt.show()
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Frequency')

    if save_path is not None:
        if save_name is None:
            save_name = 'Last PI histogram'
        fig.savefig(os.path.join(save_path, save_name + '.png'), dpi=300)

    if return_ax:
        return fig, ax
    return fig

def unpack_dataframe(df, name_video, keys):
    # Filter the dataframe for the specific file name
    df_filtered = df[df['filename'].isin([name_video])]
    
    if df_filtered.empty:
        print(f'File {name_video} not found in the dataframe')
        return None, None
    
    index = df_filtered.index
    result = {}
    for key in keys:
        try:
            item = df_filtered[key].item()
            if isinstance(item, str):
                if item == 'taut' or key == 'invalid test':
                    print(f"taut or invalid test in {name_video}")
                    return None, index
                item = ast.literal_eval(item)
            if isinstance(item, float) and item.is_integer():
                item = int(item)
            result[key] = item
        except Exception as e:
            print(f"{key} not set or invalid in {name_video}: {e}")
            return None, index
    return result, index
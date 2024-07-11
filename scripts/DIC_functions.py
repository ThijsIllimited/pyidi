import pyidi
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Feature_selecter import FeatureSelecter
import pickle as pkl
import json as js
import pandas as pd

class DIC_Structure(FeatureSelecter, pyidi.pyIDI):
    def __init__(self, file_path):
        # super().__init__(file_path)
        self.file_root = os.path.dirname(file_path)
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.base_name = self.file_name.split('.')[0]
        if self.base_name[-4:-2] == '_S':
            self.base_name = self.base_name[:-4]
        self.video = pyidi.pyIDI(file_path)
        self.feature_selecter = FeatureSelecter(self.video.mraw[0])

        # Copy the attributes of the FeatureSelecter class to the DIC_Structure class
        for attr, value in self.feature_selecter.__dict__.items():
            setattr(self, attr, value)
        for attr, value in self.video.__dict__.items():
            setattr(self, attr, value)
        pass

    def plot_frame(self, sequential_image_n = 0, show_saturation = False, bit_depth = 16):
        """
        Plots a still frame from the given video object.

        Args:
            sequential_image_n (int or tuple): The index or range of sequential images to plot.
            show_saturation (bool): If True, the saturated pixels are shown in blue and green.
            bit_depth (int): The bit depth of the image.
        """
        if type(sequential_image_n) == tuple:
            still_image = np.mean(self.video.mraw[sequential_image_n[0]:sequential_image_n[1]], axis=0)
        elif isinstance(sequential_image_n, np.ndarray):
            still_image = sequential_image_n
        else:
            still_image = self.video.mraw[sequential_image_n]

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
        plt.show()
        return fig, ax
    
    def plot_path(self, points = None, d = None, d_scale = 1, sequential_image_n = 0, bit_depth = 16):
        if type(sequential_image_n) == tuple:
            still_image = np.mean(self.video.mraw[sequential_image_n[0]:sequential_image_n[1]], axis=0)
        elif isinstance(sequential_image_n, np.ndarray):
            still_image = sequential_image_n
        else:
            still_image = self.video.mraw[sequential_image_n]
        if points is None:
            points = self.points
        if d is None:
            d = self.d
        td = d_scale * d +  points.reshape(len(points),1,2)

        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.imshow(still_image, cmap='gray', vmin=0, vmax=2**bit_depth-1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if np.ndim(td) == 2:
            ax.plot(td[:,1], td[:,0], 'r-.')
        else:
            for td_point in td:
                ax.plot(td_point[:,1], td_point[:,0], 'r-', lw=1)
        ax.plot(points[:,1],points[:,0],'b*', ms=2)
        plt.show()
        return fig, ax

    def play_video(self, frame_range = None, interval=30, points = None, show_saturation = False, bit_depth = 'auto', ij_counter = (0.65, 0.05)):
        
        """
        Plays the video from the given video object.
        Args:
            frame_range (range object): The range of frames to play.
            interval (int): The interval between frames in milliseconds.
            points (ndarray): Optional tracked points to plot on the video.
            show_saturation (bool): If True, the over and undersaturated pixels are shown in green and blue, respectivel.
            bit_depth (int): The bit depth of the image.
            ij_counter (tuple): The position of the frame counter.
        """
        if bit_depth == 'auto':
            bit_depth = self.video.info['Color Bit']
        if frame_range is None:
            frame_range = range(0, self.N)
        fig, ax = plt.subplots()
        im = ax.imshow(self.video.mraw[frame_range[0]], cmap='gray')
        text = ax.text(ij_counter[0], ij_counter[1], '', transform=ax.transAxes, color='black', ha='right', va='bottom')

        if show_saturation:
            over_sat = np.where(self.video.mraw[frame_range[0]] > int(0.99*(2**bit_depth-1)))
            over_sat_plot = ax.plot(over_sat[1], over_sat[0], 'b.', alpha=0.2)
            under_sat = self.video.mraw[frame_range[0]] < int(0.01*(2**bit_depth-1))
            under_sat_plot = ax.plot(under_sat[1], under_sat[0], 'g.', alpha=0.2)
        if points is not None:
            pts = ax.plot(points[:,0,1], points[:,0,0], 'r.')

        def update(i):
            im.set_data(self.video.mraw[i])
            text.set_text(f'Frame {i}')
            if show_saturation:
                over_sat = np.where(self.video.mraw[i] > int(0.99*(2**bit_depth-1)))
                over_sat_plot[0].set_data(over_sat[1], over_sat[0])
                under_sat = np.where(self.video.mraw[i] < int(0.01*(2**bit_depth-1)))
                under_sat_plot[0].set_data(under_sat[1], under_sat[0])
            if points is not None:
                pts[0].set_data(points[:,i,1], points[:,i,0])
            return im, text

        ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=interval)
        plt.show()
        return ani
    
    def load_results(self, test_number = 0):
        """
        Opens the results file and loads the data into the object.
        Args:
            path (str): The path to the results file.
        """
        previous_files_root = os.path.join(self.file_root, self.file_name.split('.')[0] + '_pyidi_analysis', f'analysis_{str(test_number).zfill(3)}')
        with open(os.path.join(previous_files_root, 'points.pkl'), 'rb') as f:
            self.points = pkl.load(f)
        with open(os.path.join(previous_files_root, 'results.pkl'), 'rb') as f:
            self.d = pkl.load(f)
        with open(os.path.join(previous_files_root, 'settings.txt'), 'rb') as f:
            self.settings = js.load(f)
            # self.settings = f.read().decode('utf-8')
        return self.points, self.d, self.settings
    
    def open_impact_data(self, root_impact):
        files = os.listdir(root_impact)
        for file in files:
            if self.base_name in file[16:-4]:
                path = os.path.join(root_impact, file)
                with open(path, 'rb') as f:
                    self.impact_data = pkl.load(f)
                break
        else:
            print(f"{self.base_name} is not in files")
            return

    def load_settings(self, test_number = 0):
        """
        Opens the results file and loads the data into the object.
        Args:
            path (str): The path to the results file.
        """
        previous_files_root = os.path.join(self.file_root, self.file_name.split('.')[0] + '_pyidi_analysis', f'analysis_{str(test_number).zfill(3)}')
        with open(os.path.join(previous_files_root, 'settings.txt'), 'rb') as f:
            self.settings = js.load(f)
        return self.settings
    
    def list_test_data(self, test_range = range(1,10), max_d = 10, max_dd = 1, max_drift = False, d_min=False, robostness_check = True):
        """
        Lists the test data in the results directory.
        Args:
            max_d (float): The maximum displacement to consider a point as tracked.
            max_dd (float): The maximum second derivative of the displacement to consider a point as tracked.
            test_range (range): The range of test numbers to consider.
        """
        df = {'cih_file':[], 'test_number':[], 'createdate':[], 'method':[], 'roi_size':[], 'n_points':[], 'n_tracked_points':[], 'success_rate': [], 'dyx':[], 'smoothing_size':[]}
        for test_number in test_range:
            if robostness_check:
                points, d, settings = self.load_results(test_number=test_number)
                td = d +  points.reshape(len(points),1,2)
                mask = np.ones(len(points), dtype=bool)
                if max_d:
                    mask = mask & ((np.max(np.linalg.norm(d, axis=2), axis = 1) < max_d))
                if max_dd:
                    mask = mask & (np.max(np.linalg.norm(np.diff(d, axis = 1), axis=2), axis = 1) < max_dd)
                if max_drift:
                    mask = mask & np.abs(np.mean(np.linalg.norm(d[:,:-100], axis=2), axis=1) < max_drift)
                if d_min:
                    mask = mask & np.max(np.linalg.norm(d, axis=2), axis = 1) > d_min
                
                n_tracked_points = np.sum(mask)
                df['n_tracked_points'].append(n_tracked_points)
                df['n_points'].append(len(points))
                df['success_rate'].append(n_tracked_points/len(points))
            else:
                settings = self.load_settings(test_number=test_number)
                df['n_tracked_points'].append(None)
                df['n_points'].append(None)
                df['success_rate'].append(None)

            cih_file    = settings['cih_file']
            createdate  = settings['createdate']
            self.info = settings['info']
            method = settings['method']
            DIC_settings = settings['settings']
            roi_size = DIC_settings['roi_size']
            try:
                smoothing_size = DIC_settings['subset_size']
            except:
                smoothing_size = None
            try :
                smoothing_size = DIC_settings['smoothing_size']
            except:
                smoothing_size = None
            try:
                polygon = np.array(self.info['Polygon'])
            except:
                polygon = None
            try:
                dyx = np.array(DIC_settings['dyx'])
            except:
                dyx = None
            df['cih_file'].append(cih_file)
            df['test_number'].append(test_number)
            df['createdate'].append(createdate)
            df['method'].append(method)
            df['roi_size'].append(roi_size)
            df['dyx'].append(dyx)
            df['smoothing_size'].append(smoothing_size)
        return pd.DataFrame(df)
    
    def join_results(self, test_number_vec = [1,2,3]):
        """
        Joins the results of multiple tests.
        Args:
            test_number_vec (list): The list of test numbers to join.
            """
        results = [self.load_results(test_number)[:2] for test_number in test_number_vec]
        points_all = np.vstack([points for points, _ in results])
        d_all = np.vstack([d for _, d in results])
        return points_all, d_all

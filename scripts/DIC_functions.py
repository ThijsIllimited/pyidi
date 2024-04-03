import pyidi
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Feature_selecter import FeatureSelecter

class DIC_Structure(FeatureSelecter, pyidi.pyIDI):
    def __init__(self, file_path):
        # super().__init__(file_path)
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

    def play_video(self, frame_range, interval=30, points = None, show_saturation = False, bit_depth = 'auto', ij_counter = (0.65, 0.05)):
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

        fig, ax = plt.subplots()
        im = ax.imshow(self.video.mraw[frame_range[0]], cmap='gray')
        text = ax.text(ij_counter[0], ij_counter[1], '', transform=ax.transAxes, color='black', ha='right', va='bottom')

        if show_saturation:
            over_sat = np.where(self.video.mraw[frame_range[0]] > int(0.99*(2**bit_depth-1)))
            over_sat_plot = ax.plot(over_sat[1], over_sat[0], 'b.', alpha=0.2)
            under_sat = self.video.mraw[frame_range[0]] < int(0.01*(2**bit_depth-1))
            under_sat_plot = ax.plot(under_sat[1], under_sat[0], 'g.', alpha=0.2)
        if points is not None:
            pts = ax.plot(points[:,0,1], points[:,0,0], '.')

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
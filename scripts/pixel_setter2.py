import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import pickle
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, binary_erosion, rotate
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_interactions import ioff, panhandler, zoom_factory
import mpl_interactions.ipyplot as iplt
from matplotlib.widgets import Slider
from skimage.feature import match_template
from PyQt5.QtWidgets import QDesktopWidget
from matplotlib.gridspec import GridSpec
import dill
from scipy.ndimage import generic_filter

ref_img = None

class PixelSetter():
    def __init__(self, image, pix_i = [], pix_j = [], file_name = None, frame_nr = None, show_methods = False, ):
        if file_name is not None:
            self.file_name  = file_name
        else:
            self.file_name  = "No file name was given"
        self.image          = image
        self.bit_depth      = 16
        self.reference_image_center_i = []
        self.reference_image_center_j = []
        self.ref_imgs       = []
        self.Low_I_limit    = 20000
        self.neighborhood_size = 3
        if image is not None:
            self.Low_I = self.image < self.Low_I_limit
        self.cor_lim_init   = 0.86
        self.sliders        = []
        self.include_rotation = False
        self.tracking_points_x = np.array([])
        self.tracking_points_y = np.array([])
        self.tracking_points_manual = []
        self.max_plots = 35
        self.y0_vec          = np.linspace(0.1, 0.9, self.max_plots)
        self.displacements = []

        self.color_vec      = ['r', 'g', 'b', 'c', 'm', 'y']
        self.marker_vec     = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
        # Initialize figures
        # Figure with Tracking dots/reference centers/ROI
        self.screen = QDesktopWidget().screenGeometry()
        self.fig_img, self.ax_img = plt.subplots()
        self.fig_img.canvas.manager.window.setGeometry(0, int(self.screen.height() * 0.1) , int(self.screen.width() * 0.6), int(self.screen.height() * 0.6))
        self.ax_img.imshow(self.image, cmap='gray')
        self.reference_fig_plot, = self.ax_img.plot([], [], 'g.', markersize=3)
        self.tracking_points_plot, = self.ax_img.plot([], [], 'r.', markersize=3, alpha=0.5)
        # Figure that shows the reference images
        self.fig_ref, self.ax_ref = plt.subplots()
        self.fig_ref.canvas.manager.window.setGeometry(int(self.screen.width() * 0.65), int(self.screen.height() * 0.10) , int(self.screen.width() * 0.3), int(self.screen.height() * 0.3))
        # Figure that shows the buttons to control Correlation Limits
        self.fig_but, self.ax_but = plt.subplots()
        self.ax_but.set_title('Control Panel')
        self.fig_but.canvas.manager.window.setGeometry(int(self.screen.width() * 0.65), int(self.screen.height() * 0.4) , int(self.screen.width() * 0.3), int(self.screen.height() * 0.6))
        self.ax_but.set_xlim(0, 1)
        self.ax_but.set_ylim(0, 1)
        self.ax_but.axis('off')
        self.fig_but.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.marker_plot = self.ax_but.plot([],[])

    def set_fig_location(self, fig, x, y, width, height):
        fig.canvas.manager.window.setGeometry(int(self.screen.width() * x), int(self.screen.height() * y) , int(self.screen.width() * width), int(self.screen.height() * height))
    
    def set_Low_I(self, Low_I_limit):
        self.Low_I_limit = Low_I_limit
        self.Low_I = self.image < self.Low_I_limit

    def set_neighborhood_size(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size
    
    def add_displacements(self, displacements):
        self.displacements = displacements

    def combine_tracking_points(self):
        tp = self.tracking_points_manual
        self.tracking_points = set(self.tracking_points_manual)
        for slider, ref_img in zip(self.sliders, self.ref_imgs):
            pix_i , pix_j = self.cross_correlate(ref_img, cor_lim=slider.val)
            self.tracking_points = self.tracking_points.union(set(zip(pix_i, pix_j)))

        self.tracking_points = list(self.tracking_points)
        
    def onclick_ref_fig(self, event):
        """Add the clicked pixels to self.reference_image_center_i and self.reference_image_center_j
        """
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if event.button == 3:
            self.ax_img.set_xlim(0, self.image.shape[1])
            self.ax_img.set_ylim(self.image.shape[0], 0)
            self.reference_image_center_i.append(y)
            self.reference_image_center_j.append(x)
            self.reference_fig_plot.set_data(self.reference_image_center_j, self.reference_image_center_i)
            self.fig_img.canvas.draw_idle()

            self.ref_imgs.append(np.array(self.image[y-self.neighborhood_size:y+(self.neighborhood_size+1), x-self.neighborhood_size:x+(self.neighborhood_size+1)]))
            self.plot_reference_images()
            self.add_slider()
            
            plt.show()
            fig_numbers = plt.get_fignums()
            if fig_numbers:
                max_fig_number = max(fig_numbers)
                plt.close(max_fig_number)
        elif event.button == 1:
            self.tracking_points_manual.append((y, x))
            # self.tracking_points_x = np.append(self.tracking_points_x, int(x))
            # self.tracking_points_y = np.append(self.tracking_points_y, int(y))
            self.tracking_points_plot.set_data(*zip(*[(y, x) for x, y in self.tracking_points_manual]))
            # self.tracking_points_plot.set_data(*zip(*self.tracking_points_manual))
            self.fig_img.canvas.draw_idle()

    def choose_reference_centers(self, include_eig_tp = False):
        zoom_factory(self.ax_img)
        if include_eig_tp:
            if not hasattr(self, 'eig_tracking_points'):
                self.tp_from_eig_img()
                print('eigenvalue tracking points were generated with default parameters')
            # self.ax_img.plot(self.eig_tracking_points[1], self.eig_tracking_points[0], 'r.', markersize=3, alpha=0.5)
            self.tracking_points_manual = [(y, x) for x, y in zip(self.eig_tracking_points[1], self.eig_tracking_points[0])]
            self.tracking_points_plot.set_data(*zip(*[(y, x) for x, y in self.tracking_points_manual]))
        self.fig_img.canvas.mpl_connect('button_press_event', self.onclick_ref_fig)
        plt.show()
        return
    
    def add_reference_image(self, pix_i, pix_j):
        self.ref_imgs.append(np.array(self.image[pix_i-self.neighborhood_size:pix_i+(self.neighborhood_size+1), pix_j-self.neighborhood_size:pix_j+(self.neighborhood_size+1)]))
    
    def cross_correlate(self, ref_img, cor_lim):
        corr = match_template(self.image, ref_img, pad_input=True)
        corr = (corr-np.min(corr)) / np.max(corr-np.min(corr))
        high_corr = corr > cor_lim
        peaks_local = self.detect_peaks(corr)
        pix_i, pix_j = np.where(peaks_local & high_corr & self.Low_I)
        return pix_i, pix_j

    def cross_correlate_x(self, cor_lim, fig_number):
        ref_img = self.ref_imgs[fig_number]
        corr_max = match_template(self.image, ref_img, pad_input=True)
        corr_max = (corr_max-np.min(corr_max)) / np.max(corr_max-np.min(corr_max))
        high_corr = corr_max > cor_lim
        peaks_local = self.detect_peaks(corr_max)
        self.tracking_points_x = np.where(peaks_local & high_corr & self.Low_I)[1]
        return self.tracking_points_x

    def cross_correlate_y(self, x, cor_lim, fig_number):
        ref_img = self.ref_imgs[fig_number]
        corr_max = match_template(self.image, ref_img, pad_input=True)
        corr_max = (corr_max-np.min(corr_max)) / np.max(corr_max-np.min(corr_max))
        high_corr = corr_max > cor_lim
        peaks_local = self.detect_peaks(corr_max)
        self.tracking_points_y = np.where(peaks_local & high_corr & self.Low_I)[0]
        return self.tracking_points_y

    def add_slider(self):
        n_plots = len(self.ref_imgs) - 1
        y0 = self.y0_vec[n_plots]
        axfreq = self.fig_but.add_axes([0.1, y0, .7, 0.05])
        self.ax_but.plot(0.9, y0*1.04+0.011, self.color_vec[n_plots%len(self.color_vec)]+self.marker_vec[n_plots%len(self.marker_vec)], markersize=10)
        slider = Slider(axfreq, label=f"cor_lim: {n_plots}", valmin=0, valmax=1.0, valinit=self.cor_lim_init)
        self.sliders.append(slider)
        controls = iplt.scatter(self.cross_correlate_x, self.cross_correlate_y, cor_lim=self.sliders[n_plots], fig_number=n_plots, ax=self.ax_img, marker=self.marker_vec[n_plots%len(self.marker_vec)],
                                     color=self.color_vec[n_plots%len(self.color_vec)], s=7)

    def plot_reference_images(self):
        plt.close(self.fig_ref)
        num_plots = len(self.ref_imgs)
        if num_plots == 1:
            self.fig_ref, self.ax_ref = plt.subplots()
            self.set_fig_location(self.fig_ref, 0.65, 0.10, 0.3, 0.3)
            self.ax_ref.imshow(self.ref_imgs[0], cmap='gray')
        elif num_plots == 2:
            self.fig_ref, self.ax_ref = plt.subplots(1, 2)
            self.set_fig_location(self.fig_ref, 0.65, 0.10, 0.3, 0.3)
            for i, ref_img in enumerate(self.ref_imgs):
                self.ax_ref[i].imshow(ref_img, cmap='gray')
        else:
            num_cols = np.ceil(np.sqrt(num_plots))
            num_rows, remainder = divmod(num_plots, num_cols)
            if remainder > 0:
                num_rows += 1
            self.fig_ref, self.ax_ref = plt.subplots(int(num_rows), int(num_cols))
            self.set_fig_location(self.fig_ref, 0.65, 0.10, 0.3, 0.3)
            for i, ref_img in enumerate(self.ref_imgs):
                row = int(i // num_cols)
                col = int(i % num_cols)
                self.ax_ref[row, col].imshow(ref_img, cmap='gray')
    
    def reset_but_fig(self):
        for i, slider in enumerate(self.sliders):
            self.cor_lim_vec[i] = slider.val
        plt.close(self.fig_but)
        self.fig_but, self.ax_but = plt.subplots()
        self.set_fig_location(self.fig_but, 0.65, 0.55, 0.3, 0.3)
        self.ax_but.set_title('Control Panel')
        self.ax_but.set_xlim(0, 1)
        self.ax_but.set_ylim(0, 1)
        self.ax_but.axis('off')
        self.fig_but.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.marker_plot = self.ax_but.plot([],[])

    def detect_peaks(self, image, neighborhood_size=2):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        # neighborhood = generate_binary_structure(2, neighborhood_size)
        neighborhood = np.ones((5,5))
        neighborhood[2,2] = 1

        #apply the local maximum filter; all pixel of maximal value 
        #in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood)==image
        #local_max is a mask that contains the peaks we are 
        #looking for, but also the background.
        #In order to isolate the peaks we must remove the background from the mask.

        #we create the mask of the background
        background = (image==0)

        #a little technicality: we must erode the background in order to 
        #successfully subtract it form local_max, otherwise a line will 
        #appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        #we obtain the final mask, containing only peaks, 
        #by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background

        return detected_peaks

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            for key, value in self.__dict__.items():
                if key == 'image':
                    value = np.array(value)
                try:
                    dill.dump((key, value), f)
                except:
                    print(f'Could not pickle {key}')
    
    def optical_gradient_scores(self, window_size):
        self.grad_score = []
        self.under_saturated_points = []
        for point in self.tracking_points:
            i, j = point
            window = self.image[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            window = np.gradient(window)
            self.grad_score.append(np.sum(window**2))
            if np.any(window < 0.01 * 2**self.bit_depth - 1):
                self.under_saturated_points.append(True)
            else:
                self.under_saturated_points.append(False)
        self.grad_score = np.array(self.grad_score)

    def find_eigenvalue_image(self, image, roi_size, method = 'direct'):
        if method == 'on_gradient':
            def filter(pixel_list):
                pixel_list = pixel_list.reshape(-1, 2)
                ATA = pixel_list.T @ pixel_list
                m = (ATA[0, 0] + ATA[1, 1])/2
                p = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
                return m - np.sqrt(m**2 - p)
            gy, gx = np.gradient(image)
            image = np.stack((gy, gx), axis=-1)
            roi_size = (roi_size, roi_size, 2)
        elif method == 'direct':
            def filter(pixel_list):
                ref_img = pixel_list.reshape(roi_size)
                gy, gx = np.gradient(ref_img)
                A = np.array([gx.flatten(), gy.flatten()]).T
                ATA = A.T @ A
                m = (ATA[0, 0] + ATA[1, 1])/2
                p = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
                return m - np.sqrt(m**2 - p)    
            roi_size = (roi_size, roi_size)
        elif method == 'direct_fast':
            def fast_gradient(image):
                """
                COPIED: source is the get_gradient function in _lukas_kanade.py from the pyidi package
                Fast gradient computation.
    
                Compute the gradient of image in both directions using central
                difference weights over 3 points.
                
                !!! WARNING:
                The edges are excluded from the analysis and the returned image
                is smaller then original.

                :param image: 2d numpy array
                :return: gradient in x and y direction
                """
                im1 = image[2:]
                im2 = image[:-2]
                Gy = (im1 - im2)/2
                im1 = image[:, 2:]
                im2 = image[:, :-2]
                Gx = (im1 - im2)/2
                return Gx[1:-1], Gy[:, 1:-1]
            
            def filter(pixel_list):
                ref_img = pixel_list.reshape(roi_size)
                gx, gy = fast_gradient(ref_img)
                A = np.array([gx.flatten(), gy.flatten()]).T
                ATA = A.T @ A
                m = (ATA[0, 0] + ATA[1, 1])/2
                p = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
                return m - np.sqrt(m**2 - p)    
            roi_size = (roi_size+2, roi_size+2)
        else:
            raise ValueError(f'Unknown method: {method}')       
        
        self.eig_img = generic_filter(image, filter, size=roi_size)
        
        if method == 'on_gradient':
            self.eig_img = self.eig_img[:,:,1]
        return self.eig_img

    def tp_from_eig_img(self, eig_img = None, size = 5, background_threshold = 0.25, local_max_treshold=0.4, exlude_sides_factor = None, show = False):
        background_pixels = (np.divide(self.image, np.max(self.image)) > background_threshold)
        if eig_img is None:
            eig_img = self.eig_img
        eig_img[background_pixels] = 0
        local_max = maximum_filter(eig_img, size=size)
        local_maxima = (eig_img == local_max)
        local_maxima[background_pixels] = False
        local_maxima[np.divide(local_max, np.max(local_max)) < local_max_treshold] = False
        if exlude_sides_factor is not None:
            height, width = self.image.shape
            i_min = int(exlude_sides_factor * height)
            i_max = int(height * (1 - exlude_sides_factor))
            j_min = int(exlude_sides_factor * width)
            j_max = int(width * (1 - exlude_sides_factor))
            local_maxima[:i_min, :] = False
            local_maxima[i_max:, :] = False
            local_maxima[:, :j_min] = False
            local_maxima[:, j_max:] = False
        self.eig_tracking_points = np.where(local_maxima)
        if show:
            fig, ax, = plt.subplots()
            # ax.imshow(self.image, cmap='gray')
            eig_max = np.nanmax(eig_img)
            ax.imshow(eig_img, cmap='gray', vmin=0.1*eig_max, vmax=eig_max)
            ax.plot(self.eig_tracking_points[1],self.eig_tracking_points[0], 'r.')
            plt.show()
            return fig, ax, self.eig_tracking_points
        return self.eig_tracking_points

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            obj = PixelSetter(np.array([]))
            while True:
                try:
                    key, value = dill.load(f)
                    obj.__dict__[key] = value
                except EOFError:
                    break
        return obj
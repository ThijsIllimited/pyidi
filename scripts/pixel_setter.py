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
from skimage.transform import resize
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import dill

ref_img = None

class PixelSetter():
    def __init__(self, image, pix_i = [], pix_j = [], file_name = None, frame_nr = None, show_methods = False):
        if file_name is not None:
            self.file_name  = file_name
        else:
            self.file_name  = "No file name was given"
        if frame_nr is not None:
            self.frame_nr   = frame_nr
        else:
            self.frame_nr   = "No frame number was given"
        self.method         = "polygon_include"
        self.image          = image
        self.clicked_points = []
        self.path           = None
        self.pix_i          = pix_i
        self.pix_j          = pix_j
        self.pix_i_selection = []
        self.pix_j_selection = []
        self.reset          = False
        self.reference_image_center_i = []
        self.reference_image_center_j = []
        self.ref_imgs       = []
        self.Low_I_limit    = 20000
        self.neighborhood_size = 3
        self.Low_I = self.image < self.Low_I_limit
        self.cor_lim_init   = 0.83
        self.cor_lim_vec    = []
        self.color_vec      = ['r', 'g', 'b', 'c', 'm', 'y']
        self.marker_vec     = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
        self.sliders        = []
        self.include_rotation = False
        self.tracking_points_vec = []
        self.tracking_points_x = []
        self.tracking_points_y = []
        self.tracking_points = set()

        # Initialize figures
        # Figure with Tracking dots/reference centers/ROI
        self.screen = QDesktopWidget().screenGeometry()
        self.fig_img, self.ax_img = plt.subplots()
        self.fig_img.canvas.manager.window.setGeometry(0, int(self.screen.height() * 0.1) , int(self.screen.width() * 0.6), int(self.screen.height() * 0.6))
        self.ax_img.imshow(self.image, cmap='gray')
        self.points_plot, = self.ax_img.plot([], [], 'go')
        self.polygon_plot, = self.ax_img.plot([], [], 'b--')
        self.ROI_plot, = self.ax_img.plot(self.pix_j, self.pix_i, 'r.', markersize=3)
        self.ROI_desel_plot, = self.ax_img.plot(self.pix_j_selection, self.pix_i_selection, 'b.', markersize=3)
        self.reference_fig_plot, = self.ax_img.plot([], [], 'g.', markersize=3)
        # Figure that shows the reference images
        self.fig_ref, self.ax_ref = plt.subplots()
        self.fig_ref.canvas.manager.window.setGeometry(int(self.screen.width() * 0.65), int(self.screen.height() * 0.10) , int(self.screen.width() * 0.3), int(self.screen.height() * 0.3))
        # Figure that shows the buttons to control Correlation Limits
        self.fig_but, self.ax_but = plt.subplots()
        self.ax_but.set_title('Control Panel')
        self.fig_but.canvas.manager.window.setGeometry(int(self.screen.width() * 0.65), int(self.screen.height() * 0.55) , int(self.screen.width() * 0.3), int(self.screen.height() * 0.3))
        self.ax_but.set_xlim(0, 1)
        self.ax_but.set_ylim(0, 1)
        self.ax_but.axis('off')
        self.fig_but.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.marker_plot = self.ax_but.plot([],[])

        if show_methods:
            self.show_methods()
    
    def set_fig_location(self, fig, x, y, width, height):
        fig.canvas.manager.window.setGeometry(int(self.screen.width() * x), int(self.screen.height() * y) , int(self.screen.width() * width), int(self.screen.height() * height))

    def set_method(self, method):
        self.method = method
        if method == "by_hand":
            self.pix_i = []
            self.pix_j = []
            self.ROI_plot.set_data([], [])
    
    def set_Low_I(self, Low_I_limit):
        self.Low_I_limit = Low_I_limit
        self.Low_I = self.image < self.Low_I_limit

    def set_neighborhood_size(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def show_methods(self):
        print("Available methods are: \n")
        print("polygon_exclude: Select a polygon to exclude from the ROI")
        print("polygon_include: Select a polygon to include in the ROI")
        print("by_hand: Select the pixels by hand")
    
    def update_tracking_points_vector(self, i):
        self.tracking_points_vec[i] = list(zip(self.tracking_points_x, self.tracking_points_y))

    def update_tracking_points(self):
        # self.tracking_points = set([t for sublist in self.tracking_points_vec for t in sublist])
        self.tracking_points = set([item for sublist in self.tracking_points_vec for item in sublist])
        # self.tracking_points = self.tracking_points | set(zip(self.tracking_points_y, self.tracking_points_x))

    def onclick(self, event):
        """
        Plot a polygon where the user clicks
        """
        if self.reset:
            self.polygon_plot.set_data([], [])
            self.points_plot.set_data([], [])
            self.clicked_points.clear()
            self.reset = False
        x, y = event.xdata, event.ydata
        self.clicked_points.append([x, y])
        xy = np.asarray(self.clicked_points)
        self.points_plot.set_data(xy[:, 0], xy[:, 1])
        if len(xy) < 2:
            return
        self.polygon_plot.set_data(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]))
        self.fig.canvas.draw_idle()
    
    def onclick_by_hand(self, event):
        """Add the clicked pixels to self.pix_i and self.pix_j
        """
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.pix_i = np.append(self.pix_i, y)
            self.pix_j = np.append(self.pix_j, x)
            self.ROI_plot.set_data(self.pix_j, self.pix_i)
        elif event.button == 3:
            self.clicked_points.append([x, y])
            xy = np.asarray(self.clicked_points)
            self.points_plot.set_data(xy[:, 0], xy[:, 1])
            if len(xy) < 2:
                return
            self.polygon_plot.set_data(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]))
            self.path = Path(np.asarray(self.clicked_points))
        self.fig.canvas.draw_idle()
        
    def onclick_ref_fig2(self, event):
        """Add the clicked pixels to self.reference_image_center_i and self.reference_image_center_j
        """
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if event.button == 3:
            self.tracking_points_vec.append([])
            self.ax_img.set_xlim(0, self.image.shape[1])
            self.ax_img.set_ylim(self.image.shape[0], 0)
            global ref_img
            self.reference_image_center_i.append(y)
            self.reference_image_center_j.append(x)
            self.reference_fig_plot.set_data(self.reference_image_center_j, self.reference_image_center_i)
            self.fig_img.canvas.draw_idle()

            self.ref_imgs.append(np.array(self.image[y-self.neighborhood_size:y+(self.neighborhood_size+1), x-self.neighborhood_size:x+(self.neighborhood_size+1)]))
            self.plot_reference_images()
            self.reset_but_fig()
            self.cross_correlate_all()
            plt.show()

    def onkey_exclude(self, event):
        """check which points are inside the polygon when the user presses 'enter'
        The points inside the selection are deleted from the ROI
        """
        if event.key == 'enter':
            self.path = Path(np.asarray(self.clicked_points))
            mask = self.path.contains_points(np.vstack((self.pix_j, self.pix_i)).T)
            self.ROI_desel_plot.set_data(self.pix_j[mask], self.pix_i[mask])
            self.pix_i_selection, self.pix_j_selection = self.pix_i[~mask], self.pix_j[~mask]
            self.ROI_plot.set_data(self.pix_j_selection, self.pix_i_selection)
            self.pix_j, self.pix_i = self.pix_j[~mask], self.pix_i[~mask]
            xy = np.asarray(self.clicked_points)
            self.ax_img.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]),'b--')
            self.polygon_plot.set_data([], [])
            self.points_plot.set_data([], [])
            self.clicked_points.clear()
            self.fig_img.canvas.draw_idle()

    def onkey_include(self, event):
        """check which points are inside the box when the user presses 'enter'
        The points inside the selection are included in the ROI
        """
        if event.key == 'enter':
            self.path = Path(np.asarray(self.clicked_points))
            mask = self.path.contains_points(np.vstack((self.pix_j, self.pix_i)).T)
            self.ROI_desel_plot.set_data(self.pix_j[~mask], self.pix_i[~mask])
            self.pix_i_selection, self.pix_j_selection = self.pix_i[mask], self.pix_j[mask]
            self.ROI_plot.set_data(self.pix_j_selection, self.pix_i_selection)
            self.pix_j, self.pix_i = self.pix_j[mask], self.pix_i[mask]

            xy = np.asarray(self.clicked_points)
            self.ax_img.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]),'b--')
            self.reset = True
            # self.polygon_plot.set_data([], [])
            # self.points_plot.set_data([], [])
            # self.clicked_points.clear()
            self.fig_img.canvas.draw_idle()
    
    def get_tracking_pixels(self):
        """
        Get the tracking pixels from the image. The user can restrict the tracking pixels to those inside a polygon. Or a user can select the tracking pixels by hand.
        """
        if self.method == "polygon_exlcude":
            self.fig_img.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig_img.canvas.mpl_connect('key_press_event', self.onkey_exclude)
            plt.show()
        elif self.method == "polygon_include":
            self.fig_img.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig_img.canvas.mpl_connect('key_press_event', self.onkey_include)
            plt.show()
        elif self.method == "by_hand":
            self.fig_img.canvas.mpl_connect('button_press_event', self.onclick_by_hand)
            plt.show()
        else:
            print("Method not implemented")
        return
    
    def plot_histogram(self, bins=200, subset_ratio=None, bits=16, inside_polygon = False, trim_factor = None, plot = True):
        if trim_factor is not None:
            trim_ids = np.floor(np.array(self.image.shape)*trim_factor).astype(int)
            still_image = self.image[trim_ids[0]:-trim_ids[0], trim_ids[1]:-trim_ids[1]]
            still_image = still_image.flatten()
        still_image = self.image.flatten()
        if subset_ratio is not None:
            still_image = np.random.choice(still_image, size=int(subset_ratio*len(still_image)), replace=True)
        if inside_polygon:
            x, y = np.meshgrid(np.arange(self.image.shape[1]), np.arange(self.image.shape[0]))
            pixels = np.vstack((x.flatten(), y.flatten())).T
            mask = self.path.contains_points(pixels)
            still_image = still_image[..., mask]
        if plot:
            plt.figure() # create a new figure
            plt.hist(still_image, bins=bins)
            plt.xlabel('Pixel intensity')
            plt.ylabel('Frequency')
            plt.title('Pixel intensity histogram')
            plt.xlim(0, 2**bits-1)
            # plt.ylim(0, 2*subset_size/bins)
            plt.show()
        return still_image

    def get_white_black_value(self, inside_polygon = True):
        still_image = self.plot_histogram(inside_polygon = inside_polygon, plot = False)
        still_image = np.sort(still_image)/np.max(still_image)
        peaks = find_peaks(still_image)
        white_value = peaks[0][0]
        black_value = peaks[0][-1]
        return white_value, black_value
    
    def choose_reference_centers(self):
        # plt.get_current_fig_manager().full_screen_toggle()
        disconnect_zoom = zoom_factory(self.ax_img)
        # Disconnect all the connections to button_press_event and key_press_event
        for cid in list(self.fig_img.canvas.callbacks.callbacks['button_press_event']):
            self.fig_img.canvas.mpl_disconnect(cid)
        for cid in list(self.fig_img.canvas.callbacks.callbacks['key_press_event']):
            self.fig_img.canvas.mpl_disconnect(cid)
        self.fig_img.canvas.mpl_connect('button_press_event', self.onclick_ref_fig2)
        plt.show()
        return
    
    def add_reference_image(self, pix_i, pix_j):
        self.ref_imgs.append(np.array(self.image[pix_i-self.neighborhood_size:pix_i+(self.neighborhood_size+1), pix_j-self.neighborhood_size:pix_j+(self.neighborhood_size+1)]))

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

    def generate_reference_image(self, neighborhood_size = 3, seperate_save = False, filename = None, path = 'C:/Users/thijs/Documents/GitHub/pyidi_data/reference_figs/', plot = False):
        self.ax_img.set_xlim(0, self.image.shape[1])
        self.ax_img.set_ylim(self.image.shape[0], 0)
        self.ref_imgs = []
        for pix_i, pix_j in zip(self.reference_image_center_i, self.reference_image_center_j):
            self.ref_imgs.append(np.array(self.image[pix_i-neighborhood_size:pix_i+(neighborhood_size+1), pix_j-neighborhood_size:pix_j+(neighborhood_size+1)]))
        if seperate_save:
            if filename is None:
                filename = self.file_name
            with open(path + filename + '.pkl', 'wb') as f:
                pickle.dump(self.ref_imgs, f)
        if plot:
            num_plots = len(self.ref_imgs)
            if num_plots == 1:
                self.fig_ref, self.ax_ref = plt.subplots()
                self.ax_ref.imshow(self.ref_imgs[0], cmap='gray')
            elif num_plots == 2:
                fig2, ax2 = plt.subplots(1, 2)
                for i, ref_img in enumerate(self.ref_imgs):
                    ax2[i].imshow(ref_img, cmap='gray')
            else:
                num_cols = np.ceil(np.sqrt(num_plots))
                num_rows, remainder = divmod(num_plots, num_cols)
                if remainder > 0:
                    num_rows += 1
                
                fig2, ax2 = plt.subplots(int(num_rows), int(num_cols))
                for i, ref_img in enumerate(self.ref_imgs):
                    row = int(i // num_cols)
                    col = int(i % num_cols)
                    ax2[row, col].imshow(ref_img, cmap='gray')

    def cross_correlate(self, ref_img, cor_lim=.75, I_lim = 20000):
        corr = match_template(self.image, ref_img, pad_input=True)
        high_corr = corr > cor_lim
        corr = (corr-np.min(corr)) / np.max(corr-np.min(corr))
        peaks_local = detect_peaks(corr)
        y, x = np.where(peaks_local & high_corr & self.Low_I)
        return x, y, corr
    
    def cross_correlate_x(self, cor_lim):
        global ref_img
        corr_max = match_template(self.image, ref_img, pad_input=True)
        if self.include_rotation:
            _ref_img = np.copy(ref_img)
            for i in range(3):
                _ref_img = np.rot90(_ref_img)
                corr = match_template(self.image, _ref_img, pad_input=True)
                corr_max = np.maximum(corr_max, corr)
        corr_max = (corr_max-np.min(corr_max)) / np.max(corr_max-np.min(corr_max))
        high_corr = corr_max > cor_lim
        peaks_local = detect_peaks(corr_max)
        self.tracking_points_x = np.where(peaks_local & high_corr & self.Low_I)[1]
        return self.tracking_points_x

    def cross_correlate_y(self, x, cor_lim):
        global ref_img
        corr_max = match_template(self.image, ref_img, pad_input=True)
        if self.include_rotation:
            _ref_img = np.copy(ref_img)
            for i in range(3):
                _ref_img = np.rot90(_ref_img)
                corr = match_template(self.image, _ref_img, pad_input=True)
                corr_max = np.maximum(corr_max, corr)
        corr_max = (corr_max-np.min(corr_max)) / np.max(corr_max-np.min(corr_max))
        high_corr = corr_max > cor_lim
        peaks_local = detect_peaks(corr_max)
        self.tracking_points_y = np.where(peaks_local & high_corr & self.Low_I)[0]
        return self.tracking_points_y

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

    def cross_correlate_all(self):
        self.reset_but_fig()
        global ref_img
        n_plots = len(self.ref_imgs)
        y0_vec = np.linspace(0.1, 0.9, n_plots)
        self.sliders = []
        self.cor_lim_vec.append(self.cor_lim_init)
        for i, (y0, ref_img) in enumerate(zip(y0_vec, self.ref_imgs)):
            axfreq = self.fig_but.add_axes([0.1, y0, .7, 0.05])
            self.marker_plot = self.ax_but.plot(0.9, y0, self.color_vec[i%len(self.color_vec)]+self.marker_vec[i%len(self.marker_vec)], markersize=10)
            
            slider = Slider(axfreq, label=f"cor_lim: {i}", valmin=0, valmax=1.01, valinit=self.cor_lim_vec[i])
            self.sliders.append(slider)
            controls = iplt.scatter(self.cross_correlate_x, self.cross_correlate_y, cor_lim=self.sliders[i], ax=self.ax_img, marker=self.marker_vec[i%len(self.marker_vec)],
                                     color=self.color_vec[i%len(self.color_vec)], s=7)
            self.update_tracking_points_vector(i)
        self.update_tracking_points()

    def generate_Zi(self, pix_i_n, pix_j_n, mu, sigma, bits, theta):
        dist = norm(mu, sigma)
        slope = np.tan(theta)
        x = np.linspace(-pix_i_n, pix_i_n, pix_i_n*12)
        y = np.linspace(-pix_i_n, pix_i_n, pix_j_n*12)

        X, Y = np.meshgrid(x, y)
        Z = np.maximum(dist.pdf(Y+slope*X), dist.pdf(-X+slope*Y))
        Z = (1- Z / np.max(Z)) * (2**bits-1)

        compressed_Z = resize(Z, (pix_i_n, pix_j_n)).astype(np.uint16)
        return X, Y, Z, compressed_Z

    def creat_synthetic_reference_pictures(self, bits=16, pix_i_n=7, pix_j_n=7, theta_vec = np.linspace(0, np.pi/4, 4), sigma=1, plot = False):
        mu = 0
        # create a normal distribution object
        n_cols = 2
        n_rows = 2
        dist        = norm(mu, sigma)
        fig, axs    = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15,15), subplot_kw={'projection': '3d'})
        fig2, axs2  = plt.subplots(ncols=2, nrows=2, figsize=(15,15))
        reference_crosses = []
        for ax_i, theta in enumerate(theta_vec):
            X, Y, Z, compressed_Z = self.generate_Zi(pix_i_n, pix_j_n, mu, sigma, bits, theta_vec[ax_i])
            reference_crosses.append(compressed_Z)
            row = ax_i // n_cols
            col = ax_i % n_cols
            axs[row, col].plot_surface(X, Y, Z, cmap='viridis')
            axs[row, col].set_xlabel('X')
            axs[row, col].set_ylabel('Y')
            axs[row, col].set_zlabel('I(x,y)')
            axs[row, col].view_init(elev=90, azim=270) # set azim to 270 degrees
            axs2[row, col].imshow(compressed_Z, cmap='gray')
        plt.show()

    def save(self, filename = None, path = 'C:/Users/thijs/Documents/GitHub/pyidi_data/tracking_pixels/'):
        if filename is None:
            filename = self.file_name
        with open(path + filename+'.pkl', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename, path = 'C:/Users/thijs/Documents/GitHub/pyidi_data/tracking_pixels/'):
        with open(path + filename+'.pkl', 'rb') as f:
            obj = dill.load(f)
        if not isinstance(obj, PixelSetter):
            raise ValueError(f"Invalid object type: {type(obj)}")
        fig, ax = plt.subplots()
        ax.imshow(obj.image, cmap='gray')
        ax.plot(obj.pix_j, obj.pix_i, 'r.', markersize=3)
        ax.plot(obj.polygon_plot.get_xdata(), obj.polygon_plot.get_ydata(), obj.polygon_plot.get_color() + obj.polygon_plot.get_linestyle())
        plt.show()
        return obj, (fig, ax)    
    
def play_video(video, frame_range, interval=30, points = None, axis = None, show_saturation = False, bit_depth = 16, color = 'r', include_W = False, roi_size = (11,11)):
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
    
    fig, ax = plt.subplots()
    im = ax.imshow(video.mraw[frame_range[0]], cmap='gray')
    text = ax.text(0.65, 0.9, '', transform=ax.transAxes, color='black', ha='right', va='bottom')
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
    plt.show()
    return ani

def detect_peaks(image, neighborhood_size=2):
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
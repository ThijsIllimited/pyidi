import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import pickle

class PixelSetter():
    def __init__(self, image, pix_i = [], pix_j = [], file_name = None, frame_nr = None, show_methods = False):
        if file_name is not None:
            self.file_name = file_name
        else:
            self.file_name = "No file name was given"
        if frame_nr is not None:
            self.frame_nr = frame_nr
        else:
            self.frame_nr = "No frame number was given"
        self.method = "polygon_include"
        self.image = image
        self.fig, self.ax = plt.subplots()
        plt.get_current_fig_manager().full_screen_toggle()
        self.ax.imshow(self.image, cmap='gray')
        self.clicked_points = []
        self.path = None
        self.points_plot, = self.ax.plot([], [], 'go')
        self.polygon_plot, = self.ax.plot([], [], 'b--')
        self.pix_i = pix_i
        self.pix_j = pix_j
        self.pix_i_selection = []
        self.pix_j_selection = []
        self.reset = False
        self.ROI_plot, = self.ax.plot(self.pix_j, self.pix_i, 'r.', markersize=3)
        self.ROI_desel_plot, = self.ax.plot(self.pix_j_selection, self.pix_i_selection, 'b.', markersize=3)
        if show_methods:
            self.show_methods()

    def set_method(self, method, reset_pixels = True):
        self.method = method
        if method == "by_hand":
            self.pix_i = []
            self.pix_j = []
            self.ROI_plot.set_data([], [])

    def show_methods(self):
        print("Available methods are: \n")
        print("polygon_exclude: Select a polygon to exclude from the ROI")
        print("polygon_include: Select a polygon to include in the ROI")
        print("by_hand: Select the pixels by hand")

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
            self.ax.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]),'b--')
            self.polygon_plot.set_data([], [])
            self.points_plot.set_data([], [])
            self.clicked_points.clear()
            self.fig.canvas.draw_idle()

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
            self.ax.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1],xy[0, 1]),'b--')
            self.reset = True
            # self.polygon_plot.set_data([], [])
            # self.points_plot.set_data([], [])
            # self.clicked_points.clear()
            self.fig.canvas.draw_idle()
    
    def get_tracking_pixels(self):
        """
        Get the tracking pixels from the image. The user can restrict the tracking pixels to those inside a polygon. Or a user can select the tracking pixels by hand.
        """
        if self.method == "polygon_exlcude":
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey_exclude)
            plt.show()
        elif self.method == "polygon_include":
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey_include)
            plt.show()
        elif self.method == "by_hand":
            self.fig.canvas.mpl_connect('button_press_event', self.onclick_by_hand)
            plt.show()
        else:
            print("Method not implemented")
        return
    
    # def set_polygon(self):
    #     self.fig.canvas.mpl_connect('button_press_event', self.onclick_by_hand)
    #     plt.show()

    
    def plot_histogram(self, bins=200, subset_ratio=None, bits=16, inside_polygon = False, trim_factor = None):
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
        plt.figure() # create a new figure
        plt.hist(still_image, bins=bins)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Frequency')
        plt.title('Pixel intensity histogram')
        plt.xlim(0, 2**bits-1)
        # plt.ylim(0, 2*subset_size/bins)
        plt.show()
    def save(self, filename = None, path = 'C:/Users/thijs/Documents/GitHub/pyidi_data/tracking_pixels/'):
        if filename is None:
            filename = self.file_name
        with open(path + filename+'.pkl', 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename, path = 'C:/Users/thijs/Documents/GitHub/pyidi_data/tracking_pixels/'):
        with open(path + filename+'.pkl', 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, PixelSetter):
            raise ValueError(f"Invalid object type: {type(obj)}")
        fig, ax = plt.subplots()
        ax.imshow(obj.image, cmap='gray')
        ax.plot(obj.pix_j, obj.pix_i, 'r.', markersize=3)
        ax.plot(obj.polygon_plot.get_xdata(), obj.polygon_plot.get_ydata(), obj.polygon_plot.get_color() + obj.polygon_plot.get_linestyle())
        plt.show()
        return obj       
    
def play_video(video, frame_range, interval=30, points = None, axis = None):
    fig, ax = plt.subplots()
    im = ax.imshow(video.mraw[frame_range[0]], cmap='gray')
    text = ax.text(0.95, 0.05, '', transform=ax.transAxes, color='black', ha='right', va='bottom')
    if points is not None:
        pts = ax.plot(points[:,0,1], points[:,0,0], 'r.')
    if axis is not None:
        ax.set_xlim(axis[0])
        ax.set_ylim(axis[1])
    # plt.plot([0, 500], [50, 50], 'r-')  
    def update(i):
        im.set_data(video.mraw[i])
        text.set_text(f'Frame {i}')
        if points is not None:
            pts[0].set_data(points[:,i,1], points[:,i,0])
            return im, text, pts
        return im, text
    
    ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=interval) # 00: range(2550,n_frames), range(3300, 3310) ,frames=range(2500, 4000)
    plt.show()
    return ani


import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter, maximum_filter
from skimage.transform import resize
from matplotlib import animation

class FeatureSelecter():
    """ Class to select features from an image using different filters.
    eig0: The smallest eigenvalue of the structure tensor.
    eig1: The largest eigenvalue of the structure tensor.
    harris: The Harris corner response function.
    trigs: The Triggs corner response function.
    harmonic_mean: The harmonic mean of the eigenvalues of the structure tensor.
    eig_theta: The eigenvalue of the structure tensor in the direction of theta. (1D DIC only)
    eig_theta_off: The eigenvalue of the structure tensor in the direction of theta + 90 degrees. (1D DIC only)

    Args:
        image (ndarray): The image to select features from.
    """
    def __init__(self, image) -> None:
        self.filter_method = 'eig0'
        self.filter = self.eig0_filter
        self.roi_size = (11, 11, 2)
        self.alpha_harris = 0.06
        self.alpha_trigs = 0.05
        self.theta = 45
        self.image = image
        self.image_shape = image.shape
        self.gi, self.gj = np.gradient(image)
        self.gx = -self.gj
        self.gy = self.gi
        self.combined_gy_gx =  np.transpose(np.array([self.gj, -self.gi]), (1, 2, 0))
        self.score_image = None
        self.parameter = None
        self.maxima = None
        self.n_points = 0
        print('Available filter methods: eig0, eig1, harris, trigs, harmonic_mean, eig_theta, eig_theta_off, eig0_test, eig1_test' )
    
    #### Setters
    def set_filter_method(self, filter_method, roi_size = None):
        if roi_size is None:
            roi_size = self.roi_size
        self.filter_method = filter_method
        if filter_method == 'eig0':
            self.filter = self.eig0_filter
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = None
        elif filter_method == 'harris':
            self.filter = self.harris_filter
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = self.alpha_harris
        elif filter_method == 'trigs':
            self.filter = self.trigs_filter
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = self.alpha_trigs
        elif filter_method == 'harmonic_mean':
            self.filter = self.harmonic_mean_filter
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = None
        elif filter_method == 'eig_theta':
            self.filter = self.eig_theta
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = np.array([np.cos(np.radians(self.theta)), np.sin(np.radians(self.theta))])
        elif filter_method == 'eig_theta_off':
            self.filter = self.eig_theta
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = np.array([np.sin(np.radians(self.theta)), -np.cos(np.radians(self.theta))])
        elif filter_method == 'eig0_test':
            self.filter = self.eig0_test
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = None
        elif filter_method == 'eig1':
            self.filter = self.eig1_filter
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = None
        elif filter_method == 'eig1_test':
            self.filter = self.eig1_test
            self.to_filter = self.combined_gy_gx
            self.set_roi_size((roi_size[0], roi_size[1], 2))
            self.parameter = None
        elif filter_method == 'Vx':
            if self.gt is None:
                print('Set gt first')
                return
            self.filter = self.Vx_filter
            self.to_filter = np.transpose(np.array([self.gx*self.gx, self.gy*self.gy, self.gx*self.gy, self.gx*self.gt, self.gy*self.gt]), (1, 2, 0))
            self.set_roi_size((roi_size[0], roi_size[1], 5))
            self.parameter = None
        elif filter_method == 'Vy':
            if self.gt is None:
                print('Set gt first')
                return
            self.filter = self.Vy_filter
            self.to_filter = np.transpose(np.array([self.gx*self.gx, self.gy*self.gy, self.gx*self.gy, self.gx*self.gt, self.gy*self.gt]), (1, 2, 0))
            self.set_roi_size((roi_size[0], roi_size[1], 5))
            self.parameter = None
        else:
            print('Filter method not found')
    
    def set_gt(self, image1, image2):
        self.gt = image2 - image1

    def set_roi_size(self, roi_size):
        self.roi_size = roi_size
    
    def set_alpha_harris(self, alpha_harris):
        self.alpha_harris = alpha_harris
    
    def set_alpha_trigs(self, alpha_trigs):
        self.alpha_trigs = alpha_trigs
    
    def set_theta(self, theta):
        self.theta = theta
    
    def set_image(self, image):
        self.image = image
        self.image_shape = image.shape
        self.Iy, self.Ix = np.gradient(image)
        self.combined_gy_gx = np.stack((self.Iy, self.Ix), axis=-1)

    #### Plotters
    def plot_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.show()
    
    def plot_gradient(self):
        plt.imshow(self.Ix, cmap='gray')
        plt.show()
        plt.imshow(self.Iy, cmap='gray')
        plt.show()
    
    def plot_score_image(self, umin=None, umax=None, maxima = False):
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        if umin is not None:
            ax.imshow(self.score_image, cmap='gray', vmin=umin, vmax=umax)
        else:
            ax.imshow(self.score_image, cmap='gray')
        if maxima is True:
            ax.scatter(self.maxima[:, 1], self.maxima[:, 0], s=5, c='r')
        plt.show()
        return fig, ax

    #### Apply filter
    def apply_filter(self, method = None, image= None, roi_size = None, downsample = 1):
        if method is not None:
            self.set_filter_method(method, self.roi_size)
        if roi_size is not None:
            self.set_filter_method(self.filter_method, roi_size)
        if image is not None:
            self.set_image(image)

        if downsample > 1:
            roi_save = self.roi_size
            self.to_filter = resize(self.combined_gy_gx, (self.image_shape[1]//downsample, self.image_shape[0]//downsample, 2))
            self.roi_size = (self.roi_size[0]//downsample, self.roi_size[1]//downsample, self.roi_size[2])

        row_of_interest = self.roi_size[-1]//2
        if self.parameter is None:
            score_image =  generic_filter(self.to_filter, self.filter, size=self.roi_size)[:, :, row_of_interest]
        
        else:
            score_image =  generic_filter(self.to_filter, self.filter, size=self.roi_size, extra_arguments = (self.parameter, ))[:, :, row_of_interest]
        score_image[np.isnan(score_image)] = 0
        self.score_image = score_image
        if downsample > 1:
            self.score_image = resize(self.score_image, (self.image_shape[0], self.image_shape[1]))
            self.roi_size = roi_save
        return self.score_image
    
    #### Pick points
    def pick_max_filter(self, score_image = None, min_distance = 5, absolute_treshold = None, threshold_percentage = 90, top_n_points = None):
        if self.score_image is None and score_image is None:
            print('Apply a filter first')
            return
        if score_image is None:
            score_image = self.score_image
        
        if absolute_treshold is not None:
            threshold = absolute_treshold
        elif top_n_points is not None:
            threshold = 0
        else:
            threshold = np.max(score_image) * threshold_percentage /100 #np.percentile(score_image, threshold_percentage)
        
        maxima = maximum_filter(score_image, size=min_distance)
        maxima = (score_image == maxima) & (score_image > threshold)
        self.maxima = np.argwhere(maxima)
        if top_n_points is not None:
            if self.maxima.shape[0] > top_n_points:
                order = np.argsort(score_image[self.maxima[:, 0], self.maxima[:, 1]])[::-1]
                self.maxima = self.maxima[order[:top_n_points]]
        self.n_points = self.maxima.shape[0]
        return self.maxima
    
    def pick_ANMS(self, score_image = None, n_points = 50, c_robust = 0.4):
        """ Adaptive Non-Maximal Suppression
        n_points: int
            Number of points to pick
        c_robust: float
            Robustness constant
        
        """
        if self.score_image is None and score_image is None:
            print('Apply a filter first')
            return
        if score_image is None:
            score_image = self.score_image
        maxima = maximum_filter(score_image, size=3)
        maxima = (score_image == maxima) & (score_image > c_robust * np.max(score_image))
        if maxima.sum() < n_points:
            self.maxima = np.argwhere(maxima)
            print('Not enough points')
            return self.maxima
        maxima = np.argwhere(maxima)
        maxima_values = score_image[maxima[:, 0], maxima[:, 1]]
        highest_value = np.max(maxima_values)
        order = np.argsort(maxima_values)
        picked_points = maxima[order[0]]
        distances = np.sum((picked_points - maxima)**2, axis=-1)
        distances_new = np.zeros_like(distances)
        points_True = np.ones_like(distances, dtype=bool)
        points_True[0] = False
        distances[0] = 0

        s = 1
        while s < n_points:
            point = maxima[points_True][np.argmax(distances[points_True])]
            distances_new[points_True]  = np.sum((point - maxima[points_True])**2, axis=-1)
            distances[points_True]      = np.minimum(distances[points_True], distances_new[points_True])
            max_distance                = np.argmax(distances)
            if maxima_values[max_distance] > c_robust * highest_value:
                picked_points = np.vstack((picked_points, point))
                s += 1
            points_True[max_distance] = False
            maxima_values[max_distance] = 0
            self.radius = np.sqrt(distances[max_distance])
            distances[max_distance] = 0

        self.maxima = picked_points
        if self.maxima.shape[0] < n_points:
            print(f'Only {s} points added.')
        
        print(f'Added {s} points with a minimum radius of {self.radius}')
        return picked_points

    def pick_max_loop(self, score_image = None, min_distance = 5, n_points = 100, minimum_score = 1):
        if self.score_image is None and score_image is None:
            print('Apply a filter first')
            return
        if score_image is None:
            score_image = self.score_image
        if type(min_distance) == int or type(min_distance) == float:
            min_distance = (min_distance, min_distance)
        # qi, qj = (min_distance[0]//2), (min_distance[1]//2)
        qi, qj = (min_distance[0]-1), (min_distance[1]-1)
        si_flat = score_image.flatten()
        score_order = np.argsort(si_flat)[::-1]
        first_low_score = np.argmax(si_flat[score_order] < minimum_score)
        score_order = score_order[:first_low_score]
        placed_points = np.zeros_like(score_image, dtype=bool)
        self.maxima = []
        for point in score_order:
            y, x = np.unravel_index(point, score_image.shape)
            if placed_points[y - qi: y + qi+1, x - qj: x + qj+1].any():
                continue
            placed_points[y, x] = True
            self.maxima.append([y, x])
            if len(self.maxima) > n_points:
                self.maxima = np.array(self.maxima[:n_points])
                break
        self.maxima = np.array(self.maxima)
        return self.maxima

    def pick_max_loop(self, S, g, n):
        si_flat = S.flatten()
        score_order = np.argsort(si_flat)[::-1]
        P = np.zeros_like(S, dtype=bool)
        features = []
        for point in score_order:
            y, x = np.unravel_index(point, S.shape)
            if P[y - g: y + g+1, x - g: x + g+1].any():
                continue
            P[y, x] = True
            features.append([y, x])
            if len(features) > n:
                features = np.array(features[:n])
                break
        return features

    #### Filter methods
    @staticmethod
    @jit(nopython=True)
    def eig0_filter(pixel_list): #eig0
        ATA00 = pixel_list[::2] @ pixel_list[::2]  # equivalent to ATA[0, 0]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]  # equivalent to ATA[0, 1]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]  # equivalent to ATA[1, 1]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return m - np.sqrt(m**2 - p)
    
    @staticmethod
    @jit(nopython=True)
    def eig1_filter(pixel_list): #eig1
        ATA00 = pixel_list[::2] @ pixel_list[::2]  # equivalent to ATA[0, 0]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]  # equivalent to ATA[0, 1]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]  # equivalent to ATA[1, 1]
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return m + np.sqrt(m**2 - p)
    
    @staticmethod
    @jit(nopython=True)
    def harris_filter(pixel_list, alpha_harris): #harris
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2] 
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        return p -  alpha_harris* (m**2) #self.alpha_harris
    
    @staticmethod
    @jit(nopython=True)
    def trigs_filter(pixel_list, alpha_trigs): #trigs
        pixel_list_even = pixel_list[::2].copy()
        pixel_list_odd = pixel_list[1::2].copy()
        ATA00 = pixel_list_even @ pixel_list_even
        ATA01 = pixel_list_even @ pixel_list_odd
        ATA11 = pixel_list_odd @ pixel_list_odd
        m = (ATA00 + ATA11) / 2
        p = ATA00 * ATA11 - ATA01**2
        temp =  np.sqrt(m**2 - p)
        lam0 = m - temp
        lam1 = m + temp
        return lam0 - lam1 * alpha_trigs
    
    @staticmethod
    @jit(nopython=True)
    def harmonic_mean_filter(pixel_list): #harmonic_mean
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        return (ATA00 * ATA11 - ATA01**2)/ (ATA00 + ATA11)
        
    @staticmethod
    @jit(nopython=True)
    def eig_theta(pixel_list, u): # eig_theta
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        ATA   = np.array([[ATA00, ATA01], [ATA01, ATA11]])
        eig_vals, eig_vecs = np.linalg.eig(ATA)
        eig_vec_0 = eig_vecs[:,0]
        eig_vec_1 = eig_vecs[:,1]
        scaled_eig_vec_0 = np.sqrt(eig_vals[0]) * eig_vec_0
        scaled_eig_vec_1 = np.sqrt(eig_vals[1]) * eig_vec_1
        scaled_eig_vec   = scaled_eig_vec_0 + scaled_eig_vec_1
        score            = np.dot(scaled_eig_vec, u)
        return score**2
    
    @staticmethod
    @jit(nopython=True)
    def eig0_test(pixel_list): # eig0_test
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        ATA   = np.array([[ATA00, ATA01], [ATA01, ATA11]])
        eig_vals1, eig_vecs1 = np.linalg.eig(ATA)
        return np.min(eig_vals1)
    
    @staticmethod
    @jit(nopython=True)
    def eig1_test(pixel_list): # eig0_test
        ATA00 = pixel_list[::2] @ pixel_list[::2]
        ATA01 = pixel_list[::2] @ pixel_list[1::2]
        ATA11 = pixel_list[1::2] @ pixel_list[1::2]
        ATA   = np.array([[ATA00, ATA01], [ATA01, ATA11]])
        eig_vals1, eig_vecs1 = np.linalg.eig(ATA)
        return np.max(eig_vals1)
    
    # @jit(nopython=True)
    @staticmethod
    def Vx_filter(pixel_list):
        IxIx    = np.sum(pixel_list[::5])
        IyIy    = np.sum(pixel_list[1::5])
        IxIy    = np.sum(pixel_list[2::5])
        IxIt    = np.sum(pixel_list[3::5])
        IyIt    = np.sum(pixel_list[4::5])
        denom = IxIx*IyIy - IxIy**2
        if denom == 0:
            return np.nan
        ATAinv = 1/denom * np.array([[IyIy, -IxIy], [-IxIy, IxIx]])
        Vx, Vy = -ATAinv @ np.array([IxIt, IyIt])
        return Vx

    @staticmethod
    @jit(nopython=True)
    def Vy_filter(pixel_list):
        IxIx    = np.sum(pixel_list[::5])
        IyIy    = np.sum(pixel_list[1::5])
        IxIy    = np.sum(pixel_list[2::5])
        IxIt    = np.sum(pixel_list[3::5])
        IyIt    = np.sum(pixel_list[4::5])
        denom = IxIx*IyIy - IxIy**2
        if denom == 0:
            return np.nan
        ATAinv = 1/denom * np.array([[IyIy, -IxIy], [-IxIy, IxIx]])
        Vx, Vy = -ATAinv @ np.array([IxIt, IyIt])
        return Vy
    
    def Directional_smoothing(self, image, roi_size, angle = 0):
        roi_size_i = np.ceil(np.cos(np.radians(self.theta)) * roi_size[0] + np.sin(np.radians(self.theta)) * roi_size[1]).astype(int)
        roi_size_j = np.ceil(np.cos(np.radians(self.theta)) * roi_size[1] + np.sin(np.radians(self.theta)) * roi_size[0]).astype(int)
        kernal = np.zeros((roi_size_i, roi_size_j))
        
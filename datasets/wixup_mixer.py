import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
import logging

class Mixer:
    log = logging.getLogger(__name__)

    def setLogLevel(self, level):
        self.log.setLevel(level)
    
    def __init__(self, cfg):
        self.WINDOW_SIZE = 512 # after discarding the other half of fft 
        self.D_RES = 3*10e8/2/4/10e9 # d_res = c/2B = 0.0375m
        self.MAX_RANGE = self.WINDOW_SIZE*self.D_RES # 512*0.0375 = 19.2m
        
        #skewness
        self.JOINT_SIZE = 0.15 #m
        self.TOLERANCE = self.JOINT_SIZE/self.D_RES #4
        # self.ALPHA = JOINT_SIZE/D_RES #4
        # self.STD_DEV = (ALPHA*100 )**0.5 #std_dev**2 ~= width
        self.ALPHA = cfg.alpha
        self.STD_DEV = cfg.std_dev #std_dev**2 ~= width

        # ablate study
        self.BOOTSTRAP = cfg.bootstrap
    
    def myskewnorm(self, x, alpha, loc, scale):
        # return norm.pdf(x, loc=loc, scale=scale)
        return skewnorm.pdf(x, alpha, loc=loc, scale=scale)
        # return norm.pdf(x, loc, scale) * norm.cdf(alpha*x)
    
    def getGaussian(self, mean, std_dev, alpha, window_size):
        """
            mean: float
            return: np.array (window_size,)
        """
        x = np.arange(window_size) # [0, n-1]
        pdf = self.myskewnorm(x, alpha, loc=mean, scale=std_dev)
        self.log.debug('{}, {}, {}, {}'.format(mean, alpha, std_dev, max(pdf)))
        return pdf
    
    def plotG(self, x, y, label, mean=0, sub=0):
        if sub:
            x, y = x[:sub], y[:sub]
        plt.plot(x, y, label=label, alpha=0.5)
        if mean:
            plt.plot([mean, mean], [0, max(y)], c='grey', alpha=0.5)
        
    def getGaussianMixture(self, mean_arr, std_dev=None, alpha=None, window_size=None):
        """
            mean_Arr: list, (n, )
            pdf: np.array, (window_size, )
            pdfs: list of np.array, (n, window_size)
        """
        assert len(mean_arr) > 0
        if not std_dev: std_dev = self.STD_DEV
        if not alpha: alpha = self.ALPHA
        if not window_size: window_size = self.WINDOW_SIZE
        
        pdfs = [self.getGaussian(m, std_dev=std_dev, alpha=alpha, window_size=window_size) for m in mean_arr]
        pdf = np.sum(np.array(pdfs), axis=0)
        # self._debug_gaussian(mean_arr, window_size, pdfs, pdf)

        assert isinstance(pdf, np.ndarray)

        return pdf, pdfs
    
    def _debug_gaussian(self, mean_arr, window_size, pdfs, pdf):
        # if self.log.getEffectiveLevel() == logging.DEBUG:
        plt.figure(figsize=(12,3))
        x = np.arange(window_size)
        sub = 200
        for i, p in enumerate(pdfs):
            self.plotG(x, p, f'{i}: {mean_arr[i]:.2f}', mean_arr[i], sub=sub)
        self.plotG(x, pdf, 'sum', sub=sub)
        plt.legend()
        plt.show()

    def euclidean_range(self, point):
        assert len(point) == 3
        return np.linalg.norm(np.array(point))
    
    def simulateRangeProfile(self, coordinates):
        """
            coordinates: np.array, (n, 3)
            pdf: np.array, (window_size, )
            pdfs: list of np.array, (n, window_size)
        """
        ranges_binIdx = [self.euclidean_range(coords)/self.D_RES for coords in coordinates]
        pdf, pdf_arr = self.getGaussianMixture(ranges_binIdx)
        return pdf, pdf_arr
    
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)  # Azimuth angle in radians
        elevation = np.arcsin(z / r)  # Elevation angle in radians
        return r, azimuth, elevation
    
    def spherical_to_cartesian(self, r, azimuth, elevation):
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        return x, y, z

    def generate_random_points(self, center, n):
        width = self.TOLERANCE
        points = (np.random.rand(n) - 0.5) * width + center
        return points.astype('int')

    def _is_intersection(self, ai, bi, aj, bj):
        diffs = [ai-bi, aj-bj]        
        # return diffs[0]*diffs[1] <0: # [0,0][0,0], [0,1][0,0] are exluded
        # avoid multi because the diffs are too small
        if diffs[0] < 0 and diffs[1] > 0: return True
        elif diffs[0] > 0 and diffs[1] < 0: return True
        else: return False

    def getIntersection(self, a, b):
        idxes = []
        n = len(a)
        assert n == len(b), 'intersection of two unequal-length arrays'
        for i in range(1, n):
            if self._is_intersection(a[i], b[i], a[i-1], b[i-1]):
                idxes += [i]
                if self.BOOTSTRAP:
                    weight = (b[i]+a[i])/2
                    weight = 1+int(weight//0.1) # at least one
                    randomness = self.generate_random_points(center=i, n=weight)
                    idxes.extend(randomness)
        return np.array(idxes)
    
    def getExp(self, weights, values):
        s = np.sum(weights) 
        if s == 0:
            return np.average(values)
        else:
            return np.dot(weights, values)/s
    
    def plot3D(self, a, b, c, lim=10):    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the first set of points in red
        labels = ['new', 'x1', 'x2'] # [a, b, c]
        C = [[c/255, 0, 0] for c in np.linspace(128, 255, len(a))] # spectrum of red
        colors = [C, 'black', 'cornflowerblue']
        for arr, label, color in zip([a, b, c], labels, colors):
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, marker='o', label=label)
        
        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        eps = 1e-16
        ax.axes.set_xlim3d(left=0.-eps, right=lim+eps)
        ax.axes.set_ylim3d(bottom=0.-eps, top=lim+eps) 
        ax.axes.set_zlim3d(bottom=0.-eps, top=lim+eps) 
        ax.azim = 45
        plt.legend()
        plt.show()
    
    def getNewAngles(self, idx, coordinates, pdfs):
        n = len(coordinates)
        assert len(pdfs) == n, 'len(coordinate) == len(pdfs)'
    
        angleDistribution = [] # [[p[k][idx], a[k], z[k]], ...n],  n = len(n_coords1+n_coords2)
        for k in range(n):
            _, a, e = self.cartesian_to_spherical(*coordinates[k])
            angleDistribution.append([pdfs[k][idx], a, e]) # given (p, a, e) from previous points
        angleDistribution = np.array(angleDistribution)
        
        a = self.getExp(angleDistribution[:, 0], angleDistribution[:, 1]) # p, a
        e = self.getExp(angleDistribution[:, 0], angleDistribution[:, 2]) # p, e
        return a, e

    def removeZeroPadding(self, coordinates):
        new = [c for c in coordinates if any(c)] # exclude point [0,0,0]
        return np.array(new)
    
    def _debug_pdf(self, pdf1, pdf2):
        head = 200
        if self.log.getEffectiveLevel()==logging.DEBUG:
            plt.figure(figsize=(12, 3))
            plt.plot(pdf1[:head], alpha=0.5)
            plt.plot(pdf2[:head], alpha=0.5)
            plt.title('x1 x2 before mixing')
            plt.show()
        

    def mix(self, coordinates1, coordinates2):
        """
            coordinatesX: np.array, (n, 3)
            return: np.array, (m, 3) or None
        """
        # remove zero padding
        coordinates1 = self.removeZeroPadding(coordinates1)
        coordinates2 = self.removeZeroPadding(coordinates2)
        if len(coordinates1) == 0 or len(coordinates2) == 0:
            return np.array([])
        # transform cartesian to pdf/range profile
        pdf1, pdfs1 = self.simulateRangeProfile(coordinates1)
        pdf2, pdfs2 = self.simulateRangeProfile(coordinates2)
        # self._debug_pdf(pdf1, pdf2)

        # get new ranges by gettting intersection of two pdfs
        idxes = self.getIntersection(pdf1, pdf2)
        if len(idxes) < 1:
            return np.array([])
        ranges = idxes*self.D_RES
    
        # get expection of azimuth and elevation at each new range
        coordinates = np.concatenate((coordinates1, coordinates2), axis=0)
        pdfs = np.concatenate((pdfs1, pdfs2), axis=0)
        # for coord, p in zip(coordinates, pdfs):
        #     r, a, e = cartesian_to_spherical(*coord)
        #     self.log.debug(r/d_res, a, e, coord)

        # spherical to cartesian
        newCoords = []
        for i in range(len(idxes)):
            a, e = self.getNewAngles(idxes[i], coordinates, pdfs)
            newCoords.append(self.spherical_to_cartesian(ranges[i], a, e))
        
        return np.array(newCoords)

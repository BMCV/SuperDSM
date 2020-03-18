import gocell.config  as config
import gocell.surface as surface
import numpy as np

from sklearn.cluster   import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from skimage.feature   import peak_local_max
from skimage.filters   import threshold_otsu


class PCIntensityLabels:

    def __init__(self, g, f0, f1):
        self.g  = g
        self.f0 = f0
        self.f1 = f1
    
    def __call__(self, x):
        gx = self.g(x)
        return (self.f1 - gx) ** 2 - (self.f0 - gx) ** 2
    
    def get_map(self):
        return (self.f1 - self.g.model) ** 2 - (self.f0 - self.g.model) ** 2


def _reverse_surface_intensities(g):
    rg = surface.Surface(g.model.shape)
    rg.model = g.model.max() - g.model
    rg.mask  = g.mask
    return rg


class ThresholdedLabels:

    def __init__(self, g, threshold):
        self.g = g
        self.t = threshold
    
    def __call__(self, x):
        gx = self.g(x)
        return self.g(x) - self.t
    
    def get_map(self):
        return self.g.model - self.t

    @staticmethod
    def compute_kde_threshold(g, bandwidth, samples, otsu_ub=True):
        """Computes the threshold based on kernel density estimate of the histogram.
        """
        assert bandwidth > 0 and samples > 0

        X_raw = g.model[g.mask].reshape(-1, 1)
        X_mean, X_std = X_raw.mean(), X_raw.std()
        X = (X_raw - X_mean) / X_std

        kde = KernelDensity(kernel    = 'gaussian',
                            bandwidth = bandwidth).fit(X)

        h = np.linspace(-1, +1, samples)
        z = np.exp(kde.score_samples(h[:, None]))
        z_minima = [t for t in peak_local_max(z.max() - z) if X_mean + h[t] * X_std > 0]

        thres_otsu  = threshold_otsu(X_raw)
        thres_kde   = np.inf if len(z_minima) == 0 else h[min(z_minima)] * X_std + X_mean

        if not otsu_ub and (X_raw > thres_kde).sum() > 0.1 * np.prod(X_raw.shape):
            return thres_kde

        if thres_kde > thres_otsu: thres_final = thres_otsu
        else:
            if (X_raw < thres_kde).sum() > 0.1 * np.prod(X_raw.shape):
                thres_final = thres_kde
            else:
                thres_final = thres_otsu
        return thres_final

    @staticmethod
    def compute_threshold(g, method='otsu', bandwidth=None, samples_count=None, extras={}):
        if method == 'otsu':
            return threshold_otsu(g.model[g.mask])
        elif method == 'kde' or method == 'kde_pure':
            otsu_ub = (method == 'kde')  ## use Otsu threshold as upperr bound if method is not pure KDE
            return ThresholdedLabels.compute_kde_threshold(g, bandwidth, samples_count, otsu_ub=otsu_ub)
        elif method == 'rkde':
            return g.model.max() - ThresholdedLabels.compute_threshold(_reverse_surface_intensities(g), 'kde', bandwidth, samples_count, extras)
        elif method == 'rkde_pure':
            return g.model.max() - ThresholdedLabels.compute_threshold(_reverse_surface_intensities(g), 'kde_pure', bandwidth, samples_count, extras)
        elif method == 'isbi':
            t_otsu, t_median = threshold_otsu(g.model[g.mask]), np.median(g.model[g.mask])
            w_otsu, w_median = config.get_value(extras, 'w_otsu', 1), config.get_value(extras, 'w_median', 1)
            return (w_otsu * t_otsu + w_median * min([t_median, t_otsu])) / (w_otsu + w_median)
        elif method == 'mean':
            return np.mean(g.model[g.mask])
        else:
            raise ValueError('unknown threshold method "%s"' % method)


class CustomLabels:

    def __init__(self, g):
        self.g = g
    
    def __call__(self, x):
        return self.g(x)
    
    def get_map(self):
        return self.g.model


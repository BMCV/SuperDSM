import numpy as np

from sklearn.cluster   import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from skimage.feature   import peak_local_max
from skimage.filter    import threshold_otsu


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
    def compute_kde_threshold(g, bandwidth, samples):
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

#        from matplotlib import pyplot as plt
#        plt.figure()
#        plt.hist(X[np.logical_and(X >= -1, X <= +1)], bins=50)
#        plt.gca().twinx().plot(h, z, 'r', lw=2)
#        for t in z_minima: plt.vlines(h[t], plt.ylim()[0], plt.ylim()[1], 'r', lw=2)
#        plt.show()

        thres_otsu  = threshold_otsu(X_raw)
        thres_kde   = np.inf if len(z_minima) == 0 else h[min(z_minima)] * X_std + X_mean

        if thres_kde > thres_otsu: thres_final = thres_otsu
        else:
            if (X_raw < thres_kde).sum() > 0.1 * np.prod(X_raw.shape):
                thres_final = thres_kde
            else:
                thres_final = thres_otsu
#        print(thres_kde)
#        print(thres_final)
        return thres_final

    @staticmethod
    def compute_threshold(g, method='otsu', bandwidth=None, samples_count=None):
        if method == 'otsu':
            return threshold_otsu(g.model[g.mask])
        elif method == 'kde':
            return ThresholdedLabels.compute_kde_threshold(g, bandwidth, samples_count)
        else:
            raise ValueError('unknown threshold method "%s"' % method)


class CustomLabels:

    def __init__(self, g):
        self.g = g
    
    def __call__(self, x):
        return self.g(x)
    
    def get_map(self):
        return self.g.model


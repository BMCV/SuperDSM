import pipeline
import config
import aux
import numpy as np

from math import sqrt, pi

from scipy                 import ndimage
from scipy.ndimage.filters import gaussian_laplace, gaussian_filter

from skimage        import morphology
from skimage.filter import threshold_otsu, rank


def find_dark_spots(g, radius, max_radius, rel_tolerance=0.2):
    h = gaussian_laplace(g, radius / sqrt(2))
    h = h / abs(h).max()
    h_labeled = ndimage.measurements.label(h > 0)[0]
    h_hist = aux.int_hist(h_labeled)
    h_means = {}
    for label, count in zip(*h_hist):
        cc = (h_labeled == label)
        if count > 2 * pi * np.square(max_radius):
            h_labeled[cc] = 0
        else:
            h_means[label] = h[cc].mean()
    t = np.mean(h_means.values()) - rel_tolerance * np.std(h_means.values())
    spot_labels = []
    for label, m in h_means.items():
        if m < t:
            h_labeled[h_labeled == label] = 0
        else:
            spot_labels.append(label)
    return h_labeled, spot_labels


def remove_dark_spots(g, radius, max_radius, rel_tolerance=0.2, smooth=2., out=None):
    out = aux.Output.get(out)
    g_spots, spot_labels = find_dark_spots(g, radius, max_radius, rel_tolerance=rel_tolerance)
    h = np.zeros_like(g)
    for label_idx, label in enumerate(spot_labels):
        cc = (g_spots == label)
        cc_radius  = sqrt(cc.sum() / (2 * pi))
        cc_dilated = morphology.dilation(cc, morphology.disk(max([1, round(cc_radius) - 1]))).astype(bool)
        g_sample0  = g[cc]
        g_sample1  = g[cc_dilated]
        h[cc] = g_sample1[g_sample1 > threshold_otsu(g_sample1)].mean() - g_sample0.mean()
        out.intermediate('Removing dark spots %d / %d' % (label_idx + 1, len(spot_labels)))
    out.write('Removed %d dark spots' % len(spot_labels))
    return g + gaussian_filter(h, smooth)


def remove_dark_spots_using_cfg(g_raw, cfg, out):
    if config.get_value(cfg, 'remove_dark_spots', False):
        g_raw = remove_dark_spots(g_raw, out=out,
                                  radius        = config.get_value(cfg, 'dark_spots_radius'       ,  8. ),
                                  max_radius    = config.get_value(cfg, 'max_dark_spots_radius'   , 15. ),
                                  rel_tolerance = config.get_value(cfg, 'dark_spots_rel_tolerance',  0.2),
                                  smooth        = config.get_value(cfg, 'dark_spots_smooth_amount',  2. ))
    return g_raw


def subtract_background(g_raw, smooth_amount, radius):
    g_smooth = gaussian_filter(g_raw, smooth_amount)
    g_smooth = (g_smooth * 255).round().astype('uint8')
    g_bg = rank.minimum(g_smooth, morphology.disk(radius))
    g_bg = g_bg / 255.
    return g_raw - g_bg


def subtract_background_using_cfg(g_raw, cfg, out):
    cfg = config.get_value(cfg, 'subtract_bg', False)
    if cfg == True: cfg = {}
    if isinstance(cfg, dict):
        return subtract_background(g_raw, config.get_value(cfg, 'smooth_amount', 1.), config.get_value(cfg, 'radius', 50))
    else:
        return g_raw


class Preprocessing(pipeline.Stage):

    def __init__(self):
        super(Preprocessing, self).__init__('preprocess', inputs=['g_raw'], outputs=['g_raw'])

    def process(self, input_data, cfg, out):
        g_raw = input_data['g_raw']
        g_raw =   remove_dark_spots_using_cfg(g_raw, cfg, out)
        g_raw = subtract_background_using_cfg(g_raw, cfg, out)
        return {
            'g_raw': g_raw
        }


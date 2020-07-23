import gocell.config
import gocell.pipeline

import math
import scipy.ndimage as ndi
import numpy as np


class PreprocessingStage1(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(PreprocessingStage1, self).__init__('preprocess1',
                                                  inputs  = ['g_raw'],
                                                  outputs = ['y'])

    def process(self, input_data, cfg, out, log_root_dir):
        g_raw = input_data['g_raw']
#        g_raw =   remove_dark_spots_using_cfg(g_raw, cfg, out)
#        g_raw = subtract_background_using_cfg(g_raw, cfg, out)

        sigma1 = gocell.config.get_value(cfg, 'sigma1', math.sqrt(2))
        sigma2 = gocell.config.get_value(cfg, 'sigma2', 40)
        threshold_clip = gocell.config.get_value(cfg, 'threshold_clip', np.inf)

        threshold_original = ndi.gaussian_filter(g_raw, sigma2)
        if np.isinf(threshold_clip):
            threshold_combined = threshold_original

        else:
            threshold_clip_abs = threshold_clip * g_raw.std()
            threshold_clipped  = ndi.gaussian_filter(g_raw.clip(0, threshold_clip_abs), sigma2)

            clip_area = (g_raw > threshold_clip_abs)
            _tmp1 = ndi.distance_transform_edt(~clip_area)
            _tmp1 = (sigma2 - _tmp1).clip(0, np.inf)
            _tmp1 = (_tmp1 / _tmp1.max()) ** 2
            threshold_combined = (1 - _tmp1) * threshold_clipped + _tmp1 * threshold_original

        y = ndi.gaussian_filter(g_raw, sigma1) - threshold_combined
        
        return {
            'y': y
        }


class PreprocessingStage2(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(PreprocessingStage2, self).__init__('preprocess2',
                                                  inputs  = ['y', 'seeds', 'foreground_abs_threshold'],
                                                  outputs = ['y', 'foreground_labels'])

    def process(self, input_data, cfg, out, log_root_dir):
        abs_threshold = input_data['foreground_abs_threshold']

        tmp1 = (input_data['y'] >= 0)
        tmp2 = ndi.morphology.binary_dilation(tmp1)
        tmp3 = ndi.label(tmp2)[0]
        tmp4 = np.zeros_like(tmp3)

        for seed in input_data['seeds']:
            seed = tuple(seed)
            l = tmp3[seed]
            cc = (tmp3 == l)
            tmp4[cc] = l

        foreground_labels = tmp4

        tmp10 = np.logical_and(input_data['y'] > 0, input_data['y'] < abs_threshold * input_data['y'].max())
        tmp10 = np.logical_and(tmp10, foreground_labels == 0)
        input_data['y'][tmp10] = 0

        return {
            'foreground_labels': foreground_labels,
            'y': input_data['y']
        }


#import gocell.pipeline as pipeline
#import gocell.config   as config
#import gocell.aux      as aux
#import numpy as np
#
#from math import sqrt, pi
#
#from scipy                 import ndimage
#from scipy.ndimage.filters import gaussian_laplace, gaussian_filter
#
#from skimage         import morphology
#from skimage.filters import threshold_otsu, rank
#
#
#def find_dark_spots(g, radius, max_radius, rel_tolerance=0.2):
#    h = gaussian_laplace(g, radius / sqrt(2))
#    h = h / abs(h).max()
#    h_labeled = ndimage.measurements.label(h > 0)[0]
#    h_hist = aux.int_hist(h_labeled)
#    h_means = {}
#    for label, count in zip(*h_hist):
#        cc = (h_labeled == label)
#        if count > 2 * pi * np.square(max_radius):
#            h_labeled[cc] = 0
#        else:
#            h_means[label] = h[cc].mean()
#    h_means_values = list(h_means.values())
#    t = np.mean(h_means_values) - rel_tolerance * np.std(h_means_values)
#    spot_labels = []
#    for label, m in h_means.items():
#        if m < t:
#            h_labeled[h_labeled == label] = 0
#        else:
#            spot_labels.append(label)
#    return h_labeled, spot_labels
#
#
#def remove_dark_spots(g, radius, max_radius, rel_tolerance=0.2, smooth=2., out=None):
#    out = aux.Output.get(out)
#    g_spots, spot_labels = find_dark_spots(g, radius, max_radius, rel_tolerance=rel_tolerance)
#    h = np.zeros_like(g)
#    for label_idx, label in enumerate(spot_labels):
#        cc = (g_spots == label)
#        cc_radius  = sqrt(cc.sum() / (2 * pi))
#        cc_dilated = morphology.dilation(cc, morphology.disk(max([1, round(cc_radius) - 1]))).astype(bool)
#        g_sample0  = g[cc]
#        g_sample1  = g[cc_dilated]
#        h[cc] = g_sample1[g_sample1 > threshold_otsu(g_sample1)].mean() - g_sample0.mean()
#        out.intermediate('Removing dark spots %d / %d' % (label_idx + 1, len(spot_labels)))
#    out.write('Removed %d dark spots' % len(spot_labels))
#    return g + gaussian_filter(h, smooth)
#
#
#def remove_dark_spots_using_cfg(g_raw, cfg, out):
#    if config.get_value(cfg, 'remove_dark_spots', False):
#        g_raw = remove_dark_spots(g_raw, out=out,
#                                  radius        = config.get_value(cfg, 'dark_spots_radius'       ,  8. ),
#                                  max_radius    = config.get_value(cfg, 'max_dark_spots_radius'   , 15. ),
#                                  rel_tolerance = config.get_value(cfg, 'dark_spots_rel_tolerance',  0.2),
#                                  smooth        = config.get_value(cfg, 'dark_spots_smooth_amount',  2. ))
#    return g_raw
#
#
#def subtract_background(g_raw, smooth_amount, radius):
#    g_smooth = gaussian_filter(g_raw, smooth_amount)
#    g_smooth = (g_smooth * 255).round().astype('uint8')
#    g_bg = rank.minimum(g_smooth, morphology.disk(radius))
#    g_bg = g_bg / 255.
#    return g_raw - g_bg
#
#
#def subtract_background_using_cfg(g_raw, cfg, out):
#    cfg = config.get_value(cfg, 'subtract_bg', False)
#    if cfg == True: cfg = {}
#    if isinstance(cfg, dict):
#        return subtract_background(g_raw, config.get_value(cfg, 'smooth_amount', 1.), config.get_value(cfg, 'radius', 50))
#    else:
#        return g_raw


import gocell.config
import gocell.pipeline

import math
import scipy.ndimage as ndi
import skimage.morphology as morph
import numpy as np


class PreprocessingStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(PreprocessingStage, self).__init__('preprocess',
                                                 inputs  = ['g_raw'],
                                                 outputs = ['y'])

    def process(self, input_data, cfg, out, log_root_dir):
        g_raw = input_data['g_raw']

        sigma1 = gocell.config.get_value(cfg, 'sigma1', math.sqrt(2))
        sigma2 = gocell.config.get_value(cfg, 'sigma2', 40)
        threshold_clip  = gocell.config.get_value(cfg, 'threshold_clip', 3)

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


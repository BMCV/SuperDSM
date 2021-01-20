import gocell.config
import gocell.pipeline
import gocell.aux

import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


def _get_local_maxima(image, footprint_radius, abs_threshold=1e-2, rel_threshold=0, max_count=np.inf):
    im_offset = image.min()
    image    -= im_offset
    image_min = ndi.minimum_filter(image, footprint=morph.disk(footprint_radius))
    image_max = ndi.maximum_filter(image, footprint=morph.disk(footprint_radius))
    image_maxima = (image == image_max)
    threshold = np.max([np.full(image.shape, abs_threshold * (im_offset + image.max())), im_offset + (1 + rel_threshold) * image_min], axis=0)
    image_maxima = np.logical_and(image_maxima, im_offset + image > threshold)
    if not np.isinf(max_count):
        significance = image_max / (1e-8 + image_min)
        max_count = int(min((max_count, image_maxima.sum())))
        if max_count > 0:
            min_significance = sorted(significance[image_maxima], reverse=True)[max_count - 1]
            image_maxima = np.logical_and(image_maxima, significance >= min_significance)
    return image_maxima, threshold


class SeedStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(SeedStage, self).__init__('find_seeds',
                                        inputs  = ['y'],
                                        outputs = ['seeds', 'foreground_threshold'])

    def process(self, input_data, cfg, out, log_root_dir):
        foreground_abs_threshold = gocell.config.get_value(cfg, 'foreground_abs_threshold',   0.01)
        foreground_rel_threshold = gocell.config.get_value(cfg, 'foreground_rel_threshold',      0)
        min_seed_distance        = gocell.config.get_value(cfg, 'min_seed_distance'       ,     30)
        max_seed_number          = gocell.config.get_value(cfg, 'max_seed_number'         , np.inf)

        seeds_map, threshold_map = _get_local_maxima(input_data['y'].copy(),
                                                     min_seed_distance,
                                                     max_count     = max_seed_number,
                                                     abs_threshold = foreground_abs_threshold,
                                                     rel_threshold = foreground_rel_threshold)

        out.write(f'Seeds: {seeds_map.sum()}')
        return {
            'seeds': np.array(list(zip(*np.where(seeds_map)))),
            'foreground_threshold': threshold_map
        }


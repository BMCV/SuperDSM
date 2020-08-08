import gocell.config
import gocell.pipeline
import gocell.aux

import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


def _get_local_maxima_detector(image, footprint_radius):
    im_offset = image.min()
    image    -= im_offset
    image_min = ndi.minimum_filter(image, footprint=morph.disk(footprint_radius))
    image_max = ndi.maximum_filter(image, footprint=morph.disk(footprint_radius))
    image_maxima = (image == image_max)
    def _get_local_maxima(abs_threshold=1e-2, max_count=np.inf, where=None):
        if max_count == 0: return np.zeros(image.shape, bool)
        _image_maxima = image_maxima if where is None else np.logical_and(image_maxima, where)
        _image_maxima = np.logical_and(_image_maxima, im_offset + image > abs_threshold * (im_offset + image.max()))
        if not np.isinf(max_count):
            significance = image_max / (1e-8 + image_min)
            max_count = int(min((max_count, _image_maxima.sum())))
            if max_count > 0:
                min_significance = sorted(significance[_image_maxima], reverse=True)[max_count - 1]
                _image_maxima = np.logical_and(_image_maxima, significance >= min_significance)
        return _image_maxima
    return _get_local_maxima


class SeedStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(SeedStage, self).__init__('find_seeds',
                                        inputs  = ['y'],
                                        outputs = ['seeds', 'foreground_abs_threshold'])

    def process(self, input_data, cfg, out, log_root_dir):
        foreground_abs_threshold = gocell.config.get_value(cfg, 'foreground_abs_threshold', 0.05)
        min_seed_distance        = gocell.config.get_value(cfg, 'min_seed_distance'       ,   20)
        max_seed_number          = gocell.config.get_value(cfg, 'max_seed_number'         ,  200)

        if not hasattr(foreground_abs_threshold, '__len__'): foreground_abs_threshold = [foreground_abs_threshold]
        else: foreground_abs_threshold = sorted(foreground_abs_threshold, reverse=True)

        seed_mask = np.ones(input_data['y'].shape, bool)
        seeds_map = np.zeros_like(seed_mask)
        _get_local_maxima = _get_local_maxima_detector(input_data['y'].copy(), min_seed_distance)
        for threshold in foreground_abs_threshold:

            out.intermediate(f'Seeding at threshold {threshold}... ({seeds_map.sum()})')
            _seeds_map = _get_local_maxima(max_count     = max_seed_number,
                                           abs_threshold = threshold,
                                           where         = seed_mask)

            seeds_map = np.logical_or(seeds_map, _seeds_map)
            unmasked_area = gocell.aux.retain_intersections(input_data['y'] > 0, _seeds_map)
            seed_mask = np.logical_and(seed_mask, ~unmasked_area)

            foreground_abs_threshold = threshold
            max_seed_number -= _seeds_map.sum()
            if max_seed_number == 0: break

        out.write(f'Seeds: {seeds_map.sum()}')
        return {
            'seeds': np.array(list(zip(*np.where(seeds_map)))),
            'foreground_abs_threshold': foreground_abs_threshold
        }


import pipeline
import config
import aux
import numpy as np
import warnings
import math

from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude

from skimage.feature import peak_local_max
from skimage         import morphology


class Seeds(pipeline.Stage):

    def __init__(self):
        super(Seeds, self).__init__('seeds', inputs=['g_raw'], outputs=['g_src', 'seeds'])

    def process(self, input_data, cfg, out):
        g_src = input_data['g_raw']

        median_radius = config.get_value(cfg, 'median_radius',  0 )
        smooth_amount = config.get_value(cfg, 'smooth_amount', 10.)

        if median_radius > 0: g_src = aux.medianf(g_src, morphology.disk(median_radius))
        if smooth_amount > 0: g_src = gaussian_filter(g_src, smooth_amount)

        seeds = peak_local_max(g_src,
                               min_distance   = config.get_value(cfg, 'min_distance'  ,     10),
                               threshold_rel  = config.get_value(cfg, 'rel_threshold' ,   1e-3),
                               num_peaks      = config.get_value(cfg, 'max_count'     , np.inf),
                               exclude_border = config.get_value(cfg, 'exclude_border',   True))

        return {
            'g_src': g_src,
            'seeds': seeds
        }


class Superpixels(pipeline.Stage):

    def __init__(self):
        super(Superpixels, self).__init__('superpixels',
                                          inputs  = ['g_src', 'seeds'],
                                          outputs = ['g_superpixels', 'g_superpixel_seeds'])

    def process(self, input_data, cfg, out):
        shape = input_data['g_src'].shape

        # Rasterize superpixel seeds
        g_superpixel_seeds = np.zeros(shape, 'uint16')
        for seed_idx, seed in enumerate(input_data['seeds']):
            g_superpixel_seeds[tuple(seed)] = g_superpixel_seeds.max() + 1

        # Apply watershed transform using image intensities
        distances = input_data['g_src'].max() - input_data['g_src']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            g_superpixels = morphology.watershed(distances, g_superpixel_seeds)
        out.write('Superpixels: %d' % g_superpixels.max())

        return {
            'g_superpixels':      g_superpixels,
            'g_superpixel_seeds': g_superpixel_seeds
        }


class SuperpixelsEntropy(pipeline.Stage):  # It is not really the entropy we compute here...

    def __init__(self):
        super(SuperpixelsEntropy, self).__init__('superpixels_entropy',
                                                 inputs  = ['g_superpixels', 'g_raw', 'g_src'],
                                                 outputs = ['g_superpixels_entropy', 'superpixels_entropies'])

    def process(self, input_data, cfg, out):
        g_raw = input_data['g_raw']
        g_src = input_data['g_src']
        g_superpixels = input_data['g_superpixels']

        g_grad_magnitude = gaussian_gradient_magnitude(g_raw, sigma=config.get_value(cfg, 'sigma', 5.))
        g_superpixels_entropy = np.zeros(g_superpixels.shape)
        superpixels_entropies = []

        for l in xrange(1, g_superpixels.max() + 1):
            superpixel = (g_superpixels == l)
            entropy    = math.sqrt(np.square(g_grad_magnitude[superpixel]).sum()) / (1 + g_src[superpixel].mean())
            g_superpixels_entropy[superpixel] = entropy
            superpixels_entropies.append(entropy)

        return {
            'g_superpixels_entropy': g_superpixels_entropy,
            'superpixels_entropies': superpixels_entropies
        }


class SuperpixelsDiscard(pipeline.Stage):

    def __init__(self):
        super(SuperpixelsDiscard, self).__init__('superpixels_discard',
                                                 inputs  = ['seeds', 'g_superpixels', 'min_region_size',
                                                            'superpixels_entropies', 'g_superpixels_entropy'],
                                                 outputs = ['g_superpixels'])

    def process(self, input_data, cfg, out):
        seeds, g_superpixels, min_region_size = input_data['seeds'], input_data['g_superpixels'], input_data['min_region_size']

        log_superpixels_entropies = np.log(input_data['superpixels_entropies'])
        entropies_abs_threshold = config.get_value(cfg, 'entropies_abs_threshold', 1e-4)
        entropies_rel_tolerance = config.get_value(cfg, 'entropies_rel_tolerance',  0.4)
        entropies_rel_threshold = math.exp(log_superpixels_entropies.mean() -
                                      entropies_rel_tolerance * log_superpixels_entropies.std())

        min_superpixel_entropy = max([entropies_abs_threshold, entropies_rel_threshold])

        total_superpixels_count, discarded_superpixels_count = g_superpixels.max(), 0
        g_superpixels = g_superpixels.copy()
        for seed_idx, seed in enumerate(seeds):
            
            seed_label = seed_idx + 1
            if input_data['g_superpixels_entropy'][tuple(seed)] < min_superpixel_entropy:
                cc = (g_superpixels == seed_label)
                
                # Never discard superpixels which are too small to form a ROI by themselfes, because
                # we cannot know whether a non-interesting superpixel belongs to foreground or background:
                if cc.sum() < min_region_size / 2: continue

                g_superpixels[cc] = 0
                discarded_superpixels_count += 1

            out.intermediate('Processed %d / %d superpixels' % (seed_label, total_superpixels_count))
        out.write('Discarded %d superpixels, %d remaining' % (discarded_superpixels_count,
                                                              total_superpixels_count - discarded_superpixels_count))

        return {
            'g_superpixels': g_superpixels
        }


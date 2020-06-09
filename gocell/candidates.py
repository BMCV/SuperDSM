import gocell.pipeline as pipeline
import gocell.config   as config
import gocell.aux      as aux
import gocell.surface  as surface
import gocell.modelfit as modelfit
import gocell.labels   as labels
import numpy as np
import collections

from gocell.preprocessing import remove_dark_spots_using_cfg, subtract_background_using_cfg

from skimage import morphology, measure
from scipy   import ndimage

from scipy.ndimage.filters import gaussian_filter
import scipy.sparse


class SuperpixelAdjacenciesGraph:
    def __init__(self, g_superpixels, out=None):
        out = aux.Output.get(out)
        self.adjacencies, se = {}, morphology.disk(1)
        for l0 in range(1, g_superpixels.max() + 1):
            cc = (g_superpixels == l0)
            cc_dilated = np.logical_and(morphology.binary_dilation(cc, se), np.logical_not(cc))
            self.adjacencies[l0] = set(g_superpixels[cc_dilated].flatten()) - {0, l0}

            out.intermediate('Processed superpixel %d / %d' % (l0, g_superpixels.max()))
        out.write('Computed superpixel adjacencies')


class SuperpixelCombinationsFactory:

    def __init__(self, adjacencies):
        self.adjacencies = adjacencies

    def expand(self, combination, accept_superpixel):
        neighbors = set([s for r in combination for s in self.adjacencies[r]])
        neighbors = [s for s in neighbors if s not in combination and accept_superpixel(s)]
        new_combinations = []
        for neighbor in neighbors:
            new_combination = frozenset(combination | {neighbor})
            if new_combination not in self.discovered_combinations:
                new_combinations.append(new_combination)
                self.discovered_combinations |= {new_combination}
        return new_combinations
    
    def find_superpixels_within_distance(self, root_superpixel, max_distance):
        """Performs breadth first search to find all superpixels within `max_distance` of `root_superpixel`.
        """
        queue  = collections.deque([(0, root_superpixel)])
        result = set([root_superpixel])
        while len(queue) > 0:
            depth, pivot = queue.popleft()
            if depth + 1 <= max_distance:
                neighbors = self.adjacencies[pivot]
                queue.extend((depth + 1, node) for node in neighbors - result)
                result |= neighbors
        return result
    
    def create_local_combinations(self, pivot_superpixel, accept_superpixel, max_depth=np.inf):
        if max_depth >= 0 and not np.isinf(max_depth):
            region = self.find_superpixels_within_distance(pivot_superpixel, max_depth)
            accept_superpixel0 = accept_superpixel
            accept_superpixel  = lambda s: s in region and accept_superpixel0(s)
        self.discovered_combinations = set([frozenset([pivot_superpixel])])
        expandable_combinations = [set([pivot_superpixel])]
        while len(expandable_combinations) > 0:
            combination = expandable_combinations.pop()
            expandable_combinations += self.expand(combination, accept_superpixel)
        return self.discovered_combinations


def count_binary_holes(mask):
    assert mask.dtype == bool or (mask.dtype == 'uint8' and mask.max() <= 1)
    bg_labels = ndimage.measurements.label(1 - mask)[0]
    return sum(1 for bg_cc_prop in measure.regionprops(bg_labels) if 0 not in bg_cc_prop.bbox[:2] and \
               mask.shape[0] != bg_cc_prop.bbox[2] and mask.shape[1] != bg_cc_prop.bbox[3])


class Candidate:
    def __init__(self):
        self.result      = None
        self.superpixels = set()
        self.energy      = np.NaN
        self.smooth_mat  = None
    
    def get_mask(self, g_superpixels):
        return np.in1d(g_superpixels, list(self.superpixels)).reshape(g_superpixels.shape)

    def get_region(self, g, g_superpixels):
        region_mask = self.get_mask(g_superpixels)
        return surface.Surface(g.model.shape, g.model, mask=region_mask)

    def copy(self):
        c = Candidate()
        if self.result is not None: c.result = self.result.copy()
        c.superpixels = set(self.superpixels)
        c.energy      = self.energy
        c.smooth_mat  = self.smooth_mat
        return c


class ComputeCandidates(pipeline.Stage):

    def __init__(self):
        super(ComputeCandidates, self).__init__('compute_candidates',
                                                inputs  = ['seeds', 'g_superpixels', 'g_superpixel_seeds', 'min_region_size'],
                                                outputs = ['candidates'])

    def process(self, input_data, cfg, out):
        candidates = []

        max_superpixel_distance = config.get_value(cfg, 'max_superpixel_distance', 60)
        max_superpixel_depth    = config.get_value(cfg, 'max_superpixel_depth'   ,  2)

        seeds              = input_data['seeds']
        g_superpixels      = input_data['g_superpixels']
        g_superpixel_seeds = input_data['g_superpixel_seeds']
        min_region_size    = input_data['min_region_size']

        superpixel_adjacencies_graph = SuperpixelAdjacenciesGraph(g_superpixels, out=out)
        for seed_label, seed in enumerate(seeds, start=1):
            if not (g_superpixels == seed_label).any(): continue

            superpixel_center = lambda s: seeds[s - 1]
            accept_superpixel = lambda s: np.linalg.norm(np.subtract(seed, superpixel_center(s))) <= max_superpixel_distance
            combinations_factory = SuperpixelCombinationsFactory(superpixel_adjacencies_graph.adjacencies)
            superpixel_combinations = \
                combinations_factory.create_local_combinations(seed_label, accept_superpixel, max_superpixel_depth)

            for superpixels in superpixel_combinations:
                candidate = Candidate()
                candidate.superpixels = superpixels
                candidate_mask = candidate.get_mask(g_superpixels)
                if candidate_mask.sum() < min_region_size: continue
                if len(superpixels) > 1 and count_binary_holes(candidate_mask) > 0: continue
                candidates.append(candidate)

            out.intermediate('Generated %d candidates from %d / %d seeds' % \
                (len(candidates), seed_label, g_superpixels.max()))

        out.write('Candidates: %d' % len(candidates))
        return {
            'candidates': candidates
        }


class FilterUniqueCandidates(pipeline.Stage):

    def __init__(self):
        super(FilterUniqueCandidates, self).__init__('filter_unique_candidates',
                                                     inputs=['candidates'], outputs=['unique_candidates'])

    def process(self, input_data, cfg, out):
        candidates = input_data['candidates']
        unique_candidates = []

        for c0i, c0 in enumerate(candidates):
            is_unique = True
            for c1 in candidates[c0i + 1:]:
                if c1.superpixels == c0.superpixels:
                    is_unique = False
                    break
            if is_unique: unique_candidates.append(c0)

            out.intermediate('Processed candidate %d / %d' % (c0i + 1, len(candidates)))

        out.write('Unique candidates: %d' % len(unique_candidates))
        return {
            'unique_candidates': unique_candidates
        }


class IntensityModels(pipeline.Stage):

    def __init__(self):
        super(IntensityModels, self).__init__('intensity_models',
                                              inputs  = ['g_raw', 'g_superpixels', 'unique_candidates'],
                                              outputs = ['intensity_thresholds', 'g'])

    def process(self, input_data, cfg, out):
        g_raw, g_superpixels, unique_candidates = input_data['g_raw'], input_data['g_superpixels'], input_data['unique_candidates']

        g_raw =   remove_dark_spots_using_cfg(g_raw, cfg, out)
        g_raw = subtract_background_using_cfg(g_raw, cfg, out)
        
        smooth_method = config.get_value(cfg, 'smooth_method', 'gaussian')
        if smooth_method == 'gaussian':
            g = surface.Surface.create_from_image(gaussian_filter(g_raw, config.get_value(cfg, 'smooth_amount', 1.)))
        elif smooth_method == 'median':
            g = surface.Surface.create_from_image(aux.medianf(g_raw, selem=morphology.disk(config.get_value(cfg, 'median_radius', 1))))
        else:
            raise ValueError('unknown smooth method: "%s"' % smooth_method)

        compute_threshold = lambda region: \
            labels.ThresholdedLabels.compute_threshold(region, method        = config.get_value(cfg, 'method'       , 'otsu'),
                                                               bandwidth     = config.get_value(cfg, 'bandwidth'    ,   0.1 ),
                                                               samples_count = config.get_value(cfg, 'samples_count',   100 ),
                                                               extras        = config.get_value(cfg, 'extras'       ,    {} ))
        intensity_thresholds = []
        pooling = config.get_value(cfg, 'pooling', 'off')
        for cidx, candidate in enumerate(unique_candidates):
            if pooling != 'off':
                thresholds = []
                for l in candidate.superpixels:
                    region_mask = (g_superpixels == l)
                    region = surface.Surface(g.model.shape, g.model, mask=region_mask)
                    thresholds.append(compute_threshold(region))
                threshold = {'min': min, 'mean': np.mean, 'median': np.median}[pooling](thresholds)
            else:
                threshold = compute_threshold(candidate.get_region(g, g_superpixels))
            intensity_thresholds.append(threshold)
            out.intermediate('Computed intensity model %d / %d' % (cidx + 1, len(unique_candidates)))
        out.write('Computed %d intensity models' % len(unique_candidates))

        return {
            'intensity_thresholds': intensity_thresholds,
            'g': g
        }


class ProcessCandidates(pipeline.Stage):

    def __init__(self, backend):
        super(ProcessCandidates, self).__init__('process_candidates',
                                                inputs  = ['g', 'g_superpixels', 'unique_candidates', 'intensity_thresholds'],
                                                outputs = ['processed_candidates'])
        self.backend = backend

    def process(self, input_data, cfg, out):
        g, g_superpixels, unique_candidates, intensity_thresholds = input_data['g'], \
                                                                    input_data['g_superpixels'], \
                                                                    input_data['unique_candidates'], \
                                                                    input_data['intensity_thresholds']

        modelfit_kwargs = {
            'epsilon':                   config.get_value(cfg, 'epsilon'                  , 1.  ),
            'rho':                       config.get_value(cfg, 'rho'                      , 1e-2),
            'w_sigma_factor':            config.get_value(cfg, 'w_sigma_factor'           , 2.  ),
            'bg_radius':                 config.get_value(cfg, 'bg_radius'                , 100 ),
            'smooth_amount':             config.get_value(cfg, 'smooth_amount'            , 10  ),
            'smooth_subsample':          config.get_value(cfg, 'smooth_subsample'         , 20  ),
            'gaussian_shape_multiplier': config.get_value(cfg, 'gaussian_shape_multiplier', 2   ),
            'sparsity_tol':              config.get_value(cfg, 'sparsity_tol'             , 0   ),
            'hessian_sparsity_tol':      config.get_value(cfg, 'hessian_sparsity_tol'     , 0   ),
            'init':                      config.get_value(cfg, 'init'                     , None),
            'cachesize':                 config.get_value(cfg, 'cachesize'                , 0   ),
            'cachetest':                 config.get_value(cfg, 'cachetest'                , None)
        }

        candidates = [c.copy() for c in unique_candidates]
        self.modelfit(g, candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out=out)
        out.write('Processed candidates: %d' % len(candidates))

        return {
            'processed_candidates': candidates
        }

    def modelfit(self, g, candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out):
        with aux.CvxoptFrame() as batch:
            batch['show_progress'] = False
            for ret_idx, ret in enumerate(self.backend(g, candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out=out)):
                candidates[ret['cidx']].result = ret['result'].map_to_image_pixels(g, ret['region'])
                candidates[ret['cidx']].energy = ret['energy']
                candidates[ret['cidx']].smooth_mat = aux.uplift_smooth_matrix(ret['smooth_mat'], ret['region'].mask)


class AnalyzeCandidates(pipeline.Stage):

    def __init__(self):
        super(AnalyzeCandidates, self).__init__('analyze_candidates',
                                                inputs  = ['g', 'g_superpixels', 'processed_candidates'],
                                                outputs = ['superpixels_covered_by'])

    def process(self, input_data, cfg, out):
        g, g_superpixels, candidates = input_data['g'], input_data['g_superpixels'], input_data['processed_candidates']
        superpixels_covered_by = {}

        x_map = g.get_map(normalized=False)
        for cidx, candidate in enumerate(candidates):
            model_fg = (candidate.result.s(x_map, candidate.smooth_mat) > 0)
            superpixels_covered_by[candidate] = candidate.superpixels & set(g_superpixels[model_fg])
            out.intermediate('Analyzed candidate %d / %d' % (cidx + 1, len(candidates)))
        out.write('Analyzed %d candidates' % len(candidates))

        return {
            'superpixels_covered_by': superpixels_covered_by
        }


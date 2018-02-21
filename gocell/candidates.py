import pipeline
import config
import aux
import numpy as np
import collections
import surface
import modelfit

from preprocessing import remove_dark_spots_using_cfg

from skimage import morphology, measure
from scipy   import ndimage

from scipy.ndimage.filters import gaussian_filter


class SuperpixelAdjacenciesGraph:
    def __init__(self, g_superpixels, out=None):
        out = aux.Output.get(out)
        self.adjacencies, se = {}, disk(1)
        for l0 in xrange(1, g_superpixels.max() + 1):
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
        return new_combination
    
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
    
    def create_local_combinations(self, pivot_superpixel, accept_superpixel, max_depth=inf):
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
        self.result              = None
        self.superpixels         = {}
        self.covered_superpixels = {}
        self.energy              = np.NaN
    
    def get_mask(self, g_superpixels):
        return np.in1d(g_superpixels, list(self.superpixels)).reshape(g_superpixels.shape)


class ComputeCandidates(pipeline.Stage):

    def __init__(self):
        super(ComputeCandidates, self).__init__('compute_candidates',
                                                inputs=['seeds', 'g_superpixels', 'g_superpixel_seeds', 'min_roi_size'],
                                                outputs=['candidates'])

    def process(self, input_data, cfg, out):
        candidates = []

        max_superpixel_distance = config.get_value(cfg, 'max_superpixel_distance', 60)
        max_superpixel_depth    = config.get_value(cfg, 'max_superpixel_depth'   ,  2)

        seeds              = input_data['seeds']
        g_superpixels      = input_data['g_superpixels']
        g_superpixel_seeds = input_data['g_superpixel_seeds']
        min_roi_size       = input_data['min_roi_size']

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
                if candidate_mask.sum() < min_roi_size: continue
                if count_binary_holes(candidate_mask) > 0: continue
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


class ProcessCandidates(pipeline.Stage):

    def __init__(self, backend):
        super(ProcessCandidates, self).__init__('process_candidates',
                                                inputs=['unique_candidates'], outputs=[])
        self.backend = backend

    def process(self, input_data, cfg, out):
        g_raw, g_superpixels, unique_candidates = input_data['g_raw'],
                                                  input_data['g_superpixels'],
                                                  input_data['unique_candidates']

        g_raw = remove_dark_spots_using_cfg(g_raw, cfg, out)
        g = surface.Surface.create_from_image(gaussian_filter(g_raw, config.get_value(cfg, 'smooth_amount', 1.)))
        unique_candidate_rois = [None] * len(unique_candidates)

        modelfit_kwargs = {
            'r_sigma': 9,  ## currently not used
            'kappa':   0,  ## currently not used
            'w_sigma_factor': config.get_value(cfg, 'w_sigma_factor', 2.)
        }

        self.modelfit(g, unique_candidates, unique_candidate_rois, g_superpixels, modelfit_kwargs, out=out)
        out.write('Processed candidates: %d' % len(unique_candidates))

        return {
            'g': g,
            'unique_candidate_rois': unique_candidate_rois
        }

    def modelfit(self, g, unique_candidates, unique_candidate_rois, g_superpixels, modelfit_kwargs, out):
        with modelfit.Frame() as batch:
            batch['show_progress'] = False
            for ret_idx, ret in enumerate(self.backend(g, unique_candidates, g_superpixels, modelfit_kwargs, out=out)):
                unique_candidates[ret['cidx']].result = ret['result']
                unique_candidates[ret['cidx']].energy = ret['energy']
                unique_candidate_rois[ret['cidx']]    = ret['roi'   ]


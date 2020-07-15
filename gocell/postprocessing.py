import gocell.pipeline as pipeline
import gocell.config   as config
import gocell.surface  as surface
import gocell.aux      as aux
import numpy as np
import math

from skimage.filters import threshold_otsu
from skimage         import morphology
from scipy           import ndimage


def threshold_accepted_energies(accepted_candidates, cfg):
    energies = np.array([c.energy for c in accepted_candidates])
    if len(energies) < 2: return np.inf
    t_otsu  = threshold_otsu(energies)
    t_gauss = energies.mean() + config.get_value(cfg, 'gauss_tolerance', 0.8) * energies.std()
    return max([t_otsu, t_gauss])


def compute_object_region_overlap(foreground_buf, candidate, g_superpixels):
    sel = candidate.fill_foreground(foreground_buf)
    roi_mask    = candidate.get_mask(g_superpixels)
    roi_overlap = np.logical_and(foreground_buf, roi_mask).sum() / float(foreground_buf.sum())
    foreground_buf[sel].fill(False)
    return roi_overlap


def compute_object_boundary(foreground_buf, candidate):
    sel = candidate.fill_foreground(foreground_buf)
    obj_interior = morphology.binary_erosion(foreground_buf, morphology.disk(1))
    obj_boundary = np.logical_xor(foreground_buf, obj_interior)
    foreground_buf[sel].fill(False)
    return obj_boundary.astype(bool)


class Postprocessing(pipeline.Stage):

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['g_raw', 'g_superpixels', 'accepted_candidates'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        g_raw, g_superpixels, accepted_candidates = input_data['g_raw'], input_data['g_superpixels'], input_data['accepted_candidates']
        rejection_causes = {}

        energy_threshold = threshold_accepted_energies(accepted_candidates, config.get_value(cfg, 'energy_threshold', {}))
        pp1_candidates = []
        for c in accepted_candidates:
            if c.energy < energy_threshold:
                pp1_candidates.append(c)
            else:
                rejection_causes[c] = 'maximum energy was %f but actual %f' % (energy_threshold, c.energy)

        foreground_buf = np.zeros(g_raw.shape, bool)
        min_obj_region_overlap = config.get_value(cfg, 'min_obj_region_overlap', 0.5)
        pp2_candidates = []
        for c in pp1_candidates:
            region_overlap = compute_object_region_overlap(foreground_buf, c, g_superpixels)
            if region_overlap >= min_obj_region_overlap:
                pp2_candidates.append(c)
            else:
                rejection_causes[c] = 'min region overlap was %f but actual %f' % (min_obj_region_overlap, region_overlap)

        r_map  = ndimage.filters.gaussian_gradient_magnitude(g_raw, config.get_value(cfg, 'r_sigma', 10.))
        r_map -= r_map.min()
        r_map /= r_map.max()

        r_map_responses = {}
        r_map_response_func = {'mean': np.mean, 'median': np.median}[config.get_value(cfg, 'boundary_func', 'mean')]
        for c in pp2_candidates:
            cc = compute_object_boundary(foreground_buf, c)
            r_map_responses[c] = r_map_response_func(r_map[cc])

        r_threshold = aux.threshold_gauss(r_map_responses.values(), mode='lower',
                                          tolerance=config.get_value(cfg, 'boundary_tolerance', np.inf))
        r_threshold = max([r_threshold,
                           max(r_map_responses.values()) * config.get_value(cfg, 'boundary_min', -np.inf)])

        pp3_candidates = []
        for c in pp2_candidates:
            r_map_response = r_map_responses[c]
            if r_map_response >= r_threshold:
                pp3_candidates.append(c)
            else:
                rejection_causes[c] = 'minimum r_map response was %f but actual %f' % (r_threshold, r_map_response)

        min_obj_radius = config.get_value(cfg, 'min_obj_radius', 0)
        max_obj_radius = config.get_value(cfg, 'max_obj_radius', np.inf)
        pp4_candidates = []
        pp4_shape = np.add(g_superpixels.shape, 2)
        for c in pp3_candidates:
            is_boundary_object = c.on_boundary
            obj_radius = math.sqrt(c.fg_fragment.sum() / math.pi)
            if obj_radius > max_obj_radius:
                rejection_causes[c] = 'radius (%s) too large (maximum is %s)' % (str(obj_radius), str(max_obj_radius))
            else:
                if not is_boundary_object and obj_radius < min_obj_radius:
                    rejection_causes[c] = 'radius (%s) too small (minimum is %s)' % (str(obj_radius), str(min_obj_radius))
                else:
                    if config.get_value(cfg, 'discard_image_boundary', False) and is_boundary_object:
                        rejection_causes[c] = 'intersects image exterior'
                    else:
                        pp4_candidates.append(c)
        
        if config.get_value(cfg, 'fill_holes', False):
            pp5_candidates = []
            for c in pp4_candidates:
                c = c.copy()
                c.fg_fragment = ndimage.morphology.binary_fill_holes(c.fg_fragment)
                pp5_candidates.append(c)
        else:
            pp5_candidates = pp4_candidates

        candidate_indices = dict(zip(accepted_candidates, range(len(accepted_candidates))))
        self.rejection_causes = dict((candidate_indices[c], cause) for c, cause in rejection_causes.items())
        return {
            'postprocessed_candidates': pp5_candidates
        }


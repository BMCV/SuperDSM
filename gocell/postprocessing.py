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


def compute_object_region_overlap(candidate, x_map, g_superpixels):
    obj_interior = candidate.result.s(x_map, candidate.smooth_mat) > 0
    roi_mask     = candidate.get_mask(g_superpixels)
    roi_overlap  = np.logical_and(obj_interior, roi_mask).sum() / float(obj_interior.sum())
    return roi_overlap


def compute_object_boundary(candidate, x_map):
    obj_foreground = candidate.result.s(x_map, candidate.smooth_mat) >= 0
    obj_interior   = morphology.binary_erosion(obj_foreground, morphology.disk(1))
    obj_boundary   = np.logical_xor(obj_foreground, obj_interior)
    return obj_boundary.astype(bool)


class Postprocessing(pipeline.Stage):

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['g_raw', 'g_superpixels', 'accepted_candidates'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out):
        g_raw, g_superpixels, accepted_candidates = input_data['g_raw'], input_data['g_superpixels'], input_data['accepted_candidates']
        self.rejection_causes = {}

        energy_threshold = threshold_accepted_energies(accepted_candidates, config.get_value(cfg, 'energy_threshold', {}))
        pp1_candidates = []
        for c in accepted_candidates:
            if c.energy < energy_threshold:
                pp1_candidates.append(c)
            else:
                self.rejection_causes[c] = 'maximum energy was %f but actual %f' % (energy_threshold, c.energy)

        min_obj_region_overlap = config.get_value(cfg, 'min_obj_region_overlap', 0.5)
        x_map = surface.get_pixel_map(g_superpixels.shape, normalized=False)
        pp2_candidates = []
        for c in pp1_candidates:
            region_overlap = compute_object_region_overlap(c, x_map, g_superpixels)
            if region_overlap >= min_obj_region_overlap:
                pp2_candidates.append(c)
            else:
                self.rejection_causes[c] = 'min region overlap was %f but actual %f' % (min_obj_region_overlap, region_overlap)

        r_map  = ndimage.filters.gaussian_gradient_magnitude(g_raw, config.get_value(cfg, 'r_sigma', 10.))
        r_map -= r_map.min()
        r_map /= r_map.max()
        self.r_map = r_map

        r_map_responses = {}
        r_map_response_func = {'mean': np.mean, 'median': np.median}[config.get_value(cfg, 'boundary_func', 'mean')]
        for c in pp2_candidates:
            cc = compute_object_boundary(c, x_map)
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
                self.rejection_causes[c] = 'minimum r_map response was %f but actual %f' % (r_threshold, r_map_response)

        min_obj_radius = config.get_value(cfg, 'min_obj_radius', 0)
        max_obj_radius = config.get_value(cfg, 'max_obj_radius', np.inf)
        pp4_candidates = []
        pp4_shape = np.add(g_superpixels.shape, 2)
        x_map_ext = surface.get_pixel_map(pp4_shape, normalized=False) - 1
        x_map_bnd_mask = np.ones(pp4_shape, bool)
        x_map_bnd_mask[1 : pp4_shape[0] - 1, 1 : pp4_shape[1] - 1] = False
        x_map_ext = x_map_ext[:, x_map_bnd_mask]
        for c in pp3_candidates:
            smooth_mat_ext = aux.uplift_smooth_matrix(c.smooth_mat.toarray(), np.logical_not(x_map_bnd_mask))[x_map_bnd_mask.reshape(-1)]
            is_boundary_object = (c.result.s(x_map_ext, smooth_mat_ext) > 0).any()
            obj_radius = math.sqrt((c.result.s(x_map, c.smooth_mat) > 0).sum() / math.pi)
            if obj_radius > max_obj_radius:
                self.rejection_causes[c] = 'radius (%s) too large (maximum is %s)' % (str(obj_radius), str(max_obj_radius))
            else:
                if not is_boundary_object and obj_radius < min_obj_radius:
                    self.rejection_causes[c] = 'radius (%s) too small (minimum is %s)' % (str(obj_radius), str(min_obj_radius))
                else:
                    if config.get_value(cfg, 'discard_image_boundary', False) and is_boundary_object:
                        self.rejection_causes[c] = 'intersects image exterior'
                    else:
                        pp4_candidates.append(c)

        return {
            'postprocessed_candidates': pp4_candidates
        }


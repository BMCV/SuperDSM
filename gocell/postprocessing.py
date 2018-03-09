import pipeline
import config
import surface
import numpy as np

from skimage.filter import threshold_otsu


def threshold_accepted_energies(accepted_candidates, cfg):
    energies = np.array([c.energy for c in accepted_candidates])
    if len(energies) < 2: return np.inf
    t_otsu  = threshold_otsu(energies)
    t_gauss = energies.mean() + config.get_value(cfg, 'gauss_tolerance', 0.8) * energies.std()
    return max([t_otsu, t_gauss])


def compute_object_region_overlap(candidate, x_map, g_superpixels):
    obj_interior = candidate.result.s(x_map) > 0
    roi_mask     = candidate.get_mask(g_superpixels)
    roi_overlap  = np.logical_and(obj_interior, roi_mask).sum() / float(obj_interior.sum())
    return roi_overlap


class Postprocessing(pipeline.Stage):

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['g_superpixels', 'accepted_candidates'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out):
        g_superpixels, accepted_candidates = input_data['g_superpixels'], input_data['accepted_candidates']
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

        return {
            'postprocessed_candidates': pp2_candidates
        }


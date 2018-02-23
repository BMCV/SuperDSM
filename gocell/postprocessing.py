import pipeline
import config
import numpy as np

from skimage.filter import threshold_otsu


def threshold_accepted_energies(accepted_candidates, cfg):
    energies = np.array([c.energy for c in accepted_candidates])
    if len(energies) < 2: return np.inf
    t_otsu  = threshold_otsu(energies)
    t_gauss = energies.mean() + config.get_value(cfg, 'gauss_tolerance', 0.8) * energies.std()
    return max([t_otsu, t_gauss])


class Postprocessing(pipeline.Stage):

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs=['accepted_candidates'],
                                             outputs=['postprocessed_candidates'])

    def process(self, input_data, cfg, out):
        accepted_candidates = input_data['accepted_candidates']

        energy_threshold = threshold_accepted_energies(accepted_candidates, config.get_value(cfg, 'energy_threshold', {}))
        postprocessed_candidates = [c for c in accepted_candidates if c.energy < energy_threshold]

        return {
            'postprocessed_candidates': postprocessed_candidates
        }


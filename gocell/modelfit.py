import aux
import labels
import mapper
import cvxopt, cvxopt.solvers
import surface
import modelfit_base
import numpy as np

from skimage.filter import threshold_otsu
from scipy          import ndimage


class Frame:

    def __enter__(self):
        self.options = aux.copy_dict(cvxopt.solvers.options)
        return self

    def __setitem__(self, key, value):
        cvxopt.solvers.options[key] = value

    def __exit__(self, *args):
        cvxopt.solvers.options.clear()
        cvxopt.solvers.options.update(self.options)


def create_region(g, region_mask):
    """Creates region of model activity at `region_mask` for image `g`.
    """
    return surface.Surface(g.model.shape, g.model, mask=region_mask)


def modelfit(region, r_sigma, kappa, w_sigma_factor, averaging, bg_radius):
    y_map = labels.ThresholdedLabels(region, threshold_otsu(region.model[region.mask])).get_map()
    w_map = modelfit_base.get_roi_weights(y_map, region, std_factor=w_sigma_factor)
    bg_mask = (ndimage.morphology.distance_transform_edt(~region.mask) < bg_radius)
    region.mask = np.logical_or(region.mask, np.logical_and(y_map < 0, bg_mask))
    w_map[~region.mask] = 0
    if averaging: w_map /= float(w_map.sum())
    J = modelfit_base.Energy(y_map, region, w_map, r_map=None, kappa=kappa)
    assert np.allclose(w_map.sum(), 1)
    return J, modelfit_base.PolynomialModel(np.array(modelfit_base.CP(J, np.random.randn(6)).solve()['x']))


def process_candidate(cidx, g, g_superpixels, candidate, modelfit_kwargs):
    region = create_region(g, candidate.get_mask(g_superpixels))
    J, result = modelfit(region, **modelfit_kwargs)
    return {
        'cidx':   cidx,
        'region': region,
        'energy': J(result),
        'result': result
    }


def fork_based_backend(num_forks):
    def _imap(g, unique_candidates, g_superpixels, modelfit_kwargs, out):
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(num_forks,
                                                                 process_candidate,
                                                                 mapper.unroll(xrange(len(unique_candidates))),
                                                                 g, g_superpixels,
                                                                 mapper.unroll(unique_candidates),
                                                                 modelfit_kwargs)):
            out.intermediate('Processed candidate %d / %d (using %d forks)' % \
                (ret_idx + 1, len(unique_candidates), num_forks))
            yield ret
    return _imap


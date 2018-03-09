import aux
import labels
import mapper
import cvxopt, cvxopt.solvers
import surface
import modelfit_base
import numpy as np

from skimage.filter import threshold_otsu
from scipy          import ndimage


def compute_r_map(g, region, r_sigma):
    r_map  = ndimage.filters.gaussian_gradient_magnitude(g.model, r_sigma)
    r_map -= r_map.min()
    r_map /= r_map.max()
    r_map[~region.mask] = 0
    r_map  = np.square(r_map)
    r_map -= r_map.min()
    r_map[r_map < threshold_otsu(r_map)] = 0
    return r_map \
        [region.offset[0] : region.offset[0] + region.model.shape[0],
         region.offset[1] : region.offset[1] + region.model.shape[1]]


def modelfit(g, region, intensity_threshold, r_sigma, kappa, w_sigma_factor, averaging, bg_radius):
    y_map = labels.ThresholdedLabels(region, intensity_threshold).get_map()
    w_map = modelfit_base.get_roi_weights(y_map, region, std_factor=w_sigma_factor)
    r_map = compute_r_map(g, region, r_sigma) if r_sigma > 0 and kappa > 0 else None
    bg_mask = (ndimage.morphology.distance_transform_edt(~region.mask) < bg_radius)
    region.mask = np.logical_or(region.mask, np.logical_and(y_map < 0, bg_mask))
    w_map[~region.mask] = 0
    if averaging: w_map /= float(w_map.sum())
    J = modelfit_base.Energy(y_map, region, w_map, r_map=r_map, kappa=kappa)
    return J, modelfit_base.PolynomialModel(np.array(modelfit_base.CP(J, np.random.randn(6)).solve()['x']))


def process_candidate(cidx, g, g_superpixels, candidate, intensity_threshold, modelfit_kwargs):
    region = candidate.get_region(g, g_superpixels)
    J, result = modelfit(g, region, intensity_threshold, **modelfit_kwargs)
    return {
        'cidx':   cidx,
        'region': region,
        'energy': J(result),
        'result': result
    }


def fork_based_backend(num_forks):
    def _imap(g, unique_candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out):
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(num_forks,
                                                                 process_candidate,
                                                                 mapper.unroll(xrange(len(unique_candidates))),
                                                                 g, g_superpixels,
                                                                 mapper.unroll(unique_candidates),
                                                                 mapper.unroll(intensity_thresholds),
                                                                 modelfit_kwargs)):
            out.intermediate('Processed candidate %d / %d (using %d forks)' % \
                (ret_idx + 1, len(unique_candidates), num_forks))
            yield ret
    return _imap


import gocell.aux           as aux
import gocell.labels        as labels
import gocell.mapper        as mapper
import gocell.surface       as surface
import gocell.modelfit_base as modelfit_base
import cvxopt, cvxopt.solvers
import numpy as np

from skimage.filters import threshold_otsu
from scipy           import ndimage


def modelfit(g, region, intensity_threshold, w_sigma_factor, bg_radius, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None):
    y_map = labels.ThresholdedLabels(region, intensity_threshold).get_map()
    w_map = modelfit_base.get_roi_weights(y_map, region, std_factor=w_sigma_factor)
    bg_mask = (ndimage.morphology.distance_transform_edt(~region.mask) < bg_radius)
    region.mask = np.logical_or(region.mask, np.logical_and(y_map < 0, bg_mask))
    w_map[~region.mask] = 0
    w_map_sum = w_map.sum()
    w_map /= float(w_map_sum)
    J = modelfit_base.Energy(y_map, region, w_map, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest}
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'gocell':
            J_gocell = modelfit_base.Energy(y_map, region, w_map, epsilon, rho, smooth_amount=np.inf, smooth_subsample=np.nan, gaussian_shape_multiplier=np.nan)
            params = modelfit_base.PolynomialModel(np.array(modelfit_base.CP(J_gocell, np.zeros(6), **CP_params).solve()['x'])).array
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    #ξ_mask = (np.abs(J.grad(params)[6:] * J.smooth_mat.shape[1]) > 1e-6)
    #J.smooth_mat = J.smooth_mat[:, ξ_mask]
    #J.p = None
    #params = params[:6 + J.smooth_mat.shape[1]]
    try:
        solution = np.array(modelfit_base.CP(J, params, **CP_params).solve()['x'])
    except ValueError as ex: # fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        solution = params    # at least something we can work with
    return w_map_sum, J, modelfit_base.PolynomialModel(solution)


def process_candidate(cidx, g, g_superpixels, candidate, intensity_threshold, modelfit_kwargs):
    modelfit_kwargs = aux.copy_dict(modelfit_kwargs)
    region    = candidate.get_region(g, g_superpixels)
    averaging = modelfit_kwargs.pop('averaging')
    factor, J, result = modelfit(g, region, intensity_threshold, **modelfit_kwargs)
    if averaging: factor = 1
    return {
        'cidx':   cidx,
        'region': region,
        'energy': factor * J(result),
        'result': result,
        'smooth_mat': J.smooth_mat
    }


def fork_based_backend(num_forks):
    def _imap(g, unique_candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out):
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(num_forks,
                                                                 process_candidate,
                                                                 mapper.unroll(range(len(unique_candidates))),
                                                                 g, g_superpixels,
                                                                 mapper.unroll(unique_candidates),
                                                                 mapper.unroll(intensity_thresholds),
                                                                 modelfit_kwargs)):
            out.intermediate('Processed candidate %d / %d (using %d forks, cache size %d)' % \
                (ret_idx + 1, len(unique_candidates), num_forks, modelfit_kwargs['cachesize']))
            yield ret
    return _imap


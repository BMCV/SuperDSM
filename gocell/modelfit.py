import gocell.aux           as aux
import gocell.labels        as labels
import gocell.mapper        as mapper
import gocell.surface       as surface
import gocell.modelfit_base as modelfit_base
import cvxopt, cvxopt.solvers
import numpy as np
import contextlib, traceback

from skimage.filters import threshold_otsu
from scipy           import ndimage


def modelfit(g, y_map, region, w_sigma_factor, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None):
    print('-- initializing --')
    w_map = modelfit_base.get_roi_weights(y_map, region, std_factor=w_sigma_factor)
    w_map[~region.mask] = 0
    w_map_sum = w_map.sum()
    w_map /= float(w_map_sum)
    smooth_matrix_factory = modelfit_base.SmoothMatrixFactory(smooth_amount, gaussian_shape_multiplier, smooth_subsample)
    J = modelfit_base.Energy(y_map, region, w_map, epsilon, rho, smooth_matrix_factory, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest}
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'gocell':
            print('-- convex programming starting: GOCELL --')
            J_gocell = modelfit_base.Energy(y_map, region, w_map, epsilon, rho, modelfit_base.SmoothMatrixFactory.NULL_FACTORY)
            params = modelfit_base.PolynomialModel(np.array(modelfit_base.CP(J_gocell, np.zeros(6), **CP_params).solve()['x'])).array
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    try:
        print('-- convex programming starting: GOCELLOS --')
        solution = np.array(modelfit_base.CP(J, params, **CP_params).solve()['x'])
    except ValueError: # fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        print('-- GOCELLOS failed: failing back to GOCELL result --')
        traceback.print_exc(file=sys.stdout)
        solution = params    # at least something we can work with
    print('-- finished --')
    return w_map_sum, smooth_matrix_factory, J, modelfit_base.PolynomialModel(solution)


def process_candidate_logged(log_root_dir, cidx, *args):
    if log_root_dir is not None:
        with open(aux.join_path(log_root_dir, f'{cidx}.txt'), 'w') as logfile:
            with contextlib.redirect_stdout(logfile):
                return process_candidate(cidx, *args)
    else:
        with contextlib.redirect_stdout(None):
            return process_candidate(cidx, *args)


def process_candidate(cidx, g, g_superpixels, candidate, intensity_threshold, modelfit_kwargs):
    modelfit_kwargs = aux.copy_dict(modelfit_kwargs)
    candidate.intensity_threshold = intensity_threshold
    candidate.bg_radius = modelfit_kwargs.pop('bg_radius')
    region, y_map = candidate.get_modelfit_region(g, g_superpixels)
    averaging = modelfit_kwargs.pop('averaging')
    factor, smooth_matrix_factory, J, result = modelfit(g, y_map, region, **modelfit_kwargs)
    if averaging: factor = 1
    candidate.energy = factor * J(result)
    candidate.result = result.map_to_image_pixels(g, region)
    candidate.smooth_mat = aux.uplift_smooth_matrix(J.smooth_mat, region.mask)
    candidate.smooth_matrix_factory = smooth_matrix_factory
    return {
        'cidx': cidx,
        'candidate': candidate
    }


def fork_based_backend(num_forks):
    def _imap(g, unique_candidates, g_superpixels, intensity_thresholds, modelfit_kwargs, out, log_root_dir):
        remaining_indices = list(range(len(unique_candidates)))
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(num_forks,
                                                                 process_candidate_logged,
                                                                 log_root_dir,
                                                                 mapper.unroll(range(len(unique_candidates))),
                                                                 g, g_superpixels,
                                                                 mapper.unroll(unique_candidates),
                                                                 mapper.unroll(intensity_thresholds),
                                                                 modelfit_kwargs)):
            remaining_indices.remove(ret['cidx'])
            suffix = ', remaining: ' + ','.join(str(i) for i in remaining_indices) if 0 < len(remaining_indices) < 10 else ''
            out.intermediate('Processed candidate %d / %d (using %d forks, cache size %d%s)' % \
                (ret_idx + 1, len(unique_candidates), num_forks, modelfit_kwargs['cachesize'], suffix))
            yield ret
    return _imap


import gocell.aux           as aux
import gocell.labels        as labels
import gocell.mapper        as mapper
import gocell.surface       as surface
import gocell.modelfit_base as modelfit_base
import cvxopt, cvxopt.solvers
import numpy as np
import contextlib, traceback, io
import ray
import sys
import multiprocessing

from skimage.filters import threshold_otsu
from scipy           import ndimage


def modelfit(g, y_map, region, scale, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, smooth_mat_max_allocations, smooth_mat_dtype, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None):
    print('-- initializing --')
    smooth_matrix_factory = modelfit_base.SmoothMatrixFactory(smooth_amount, gaussian_shape_multiplier, smooth_subsample, smooth_mat_max_allocations, smooth_mat_dtype)
    J = modelfit_base.Energy(y_map, region, epsilon, rho, smooth_matrix_factory, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest, 'scale': scale / J.smooth_mat.shape[0]}
    print(f'scale: {CP_params["scale"]:g}')
    fallback = False
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'gocell':
            print('-- convex programming starting: GOCELL --')
            J_gocell = modelfit_base.Energy(y_map, region, epsilon, rho, modelfit_base.SmoothMatrixFactory.NULL_FACTORY)
            params = modelfit_base.PolynomialModel(np.array(modelfit_base.CP(J_gocell, np.zeros(6), **CP_params).solve()['x'])).array
            print(f'solution: {J_gocell(params)}')
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    try:
        print('-- convex programming starting: GOCELLOS --')
        solution = modelfit_base.CP(J, params, **CP_params).solve()
        solution, status = np.array(solution['x']), solution['status']
        if status == 'unknown' and J(solution) > J(params):
            fallback = True  # numerical difficulties lead to a very bad solution, thus fall back to the GOCELL solution
        else:
            print(f'solution: {J(solution)}')
    except: # e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc(file=sys.stdout)
        fallback = True  # at least something we can work with
    if fallback:
        print('-- GOCELLOS failed: falling back to GOCELL result --')
        solution = params
    print('-- finished --')
    return J, modelfit_base.PolynomialModel(solution), fallback


def process_candidate_logged(log_root_dir, cidx, *args):
    if log_root_dir is not None:
        log_filename = aux.join_path(log_root_dir, f'{cidx}.txt')
        with io.TextIOWrapper(open(log_filename, 'wb', 0), write_through=True) as log_file:
            with contextlib.redirect_stdout(log_file):
                try:
                    return process_candidate(cidx, *args)
                except:
                    traceback.print_exc(file=log_file)
                    raise
    else:
        with contextlib.redirect_stdout(None):
            return process_candidate(cidx, *args)


def process_candidate(cidx, g, g_superpixels, x_map, candidate, modelfit_kwargs):
    modelfit_kwargs = aux.copy_dict(modelfit_kwargs)
    region, y_map = candidate.get_modelfit_region(g, g_superpixels, modelfit_kwargs.pop('bg_radius'))
    averaging = modelfit_kwargs.pop('averaging')
    J, result, fallback = modelfit(g, y_map, region, **modelfit_kwargs)
    padded_mask = np.pad(region.mask, 1)
    smooth_mat  = aux.uplift_smooth_matrix(J.smooth_mat, padded_mask)
    padded_foreground = (result.map_to_image_pixels(g, region, pad=1).s(x_map, smooth_mat) > 0)
    foreground = padded_foreground[1:-1, 1:-1]
    if foreground.any():
        rows = foreground.any(axis=1)
        cols = foreground.any(axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        candidate.fg_offset   = np.array([rmin, cmin])
        candidate.fg_fragment = foreground[rmin : rmax + 1, cmin : cmax + 1]
    else:
        candidate.fg_offset   = np.zeros(2, int)
        candidate.fg_fragment = np.zeros((1, 1), bool)
    candidate.energy      = J(result)
    candidate.on_boundary = padded_foreground[0].any() or padded_foreground[-1].any() or padded_foreground[:, 0].any() or padded_foreground[:, -1].any()
    return {
        'cidx': cidx,
        'candidate': candidate
    }


@ray.remote
def _ray_process_candidate(*args):
    return process_candidate_logged(*args)


def serial_backend():
    def map(g, unique_candidates, g_superpixels, modelfit_kwargs, out, log_root_dir):
        n = len(unique_candidates)
        x_map = g.get_map(normalized=False, pad=1)
        for cidx in range(n):
            ret = process_candidate_logged(log_root_dir, cidx, g, g_superpixels, x_map, unique_candidates[cidx], modelfit_kwargs)
            out.intermediate('Processed candidate %d / %d (cache size %d)' % (cidx + 1, n, modelfit_kwargs['cachesize']))
            yield ret
    return map


def ray_based_backend():
    def _imap(g, unique_candidates, g_superpixels, modelfit_kwargs, out, log_root_dir):
        n = len(unique_candidates)
        remaining_indices = list(range(n))
        g_id = ray.put(g)
        g_superpixels_id = ray.put(g_superpixels)
        x_map_id = ray.put(g.get_map(normalized=False, pad=1))
        try:
            futures = [_ray_process_candidate.remote(log_root_dir, cidx, g_id, g_superpixels_id, x_map_id, unique_candidates[cidx], modelfit_kwargs) for cidx in range(n)]
            for ret_idx, ret in enumerate(aux.get_ray_1by1(futures)):
                remaining_indices.remove(ret['cidx'])
                out.intermediate('Processed candidate %d / %d (cache size %d)' % (ret_idx + 1, n, modelfit_kwargs['cachesize']))
                yield ret
        except:
            pass
            out.write('Remaining candidates: ' + ((','.join(str(i) for i in remaining_indices) if len(remaining_indices) > 0 else 'None')))
            raise
    return _imap


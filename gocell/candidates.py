import gocell.aux
import gocell.surface
import gocell.modelfit

import ray
import sys, io, contextlib, traceback, time
import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


class Candidate:
    def __init__(self):
        self.fg_offset       = None
        self.fg_fragment     = None
        self.footprint       = set()
        self.energy          = np.nan
        self.on_boundary     = np.nan
        self.processing_time = np.nan
    
    def get_mask(self, g_atoms):
        return np.in1d(g_atoms, list(self.footprint)).reshape(g_atoms.shape)

    def get_region(self, g, g_superpixels):
        return g.get_region(self.get_mask(g_superpixels))

    def set(self, state):
        self.fg_fragment     = state.fg_fragment.copy() if state.fg_fragment is not None else None
        self.fg_offset       = state.fg_offset.copy() if state.fg_offset is not None else None
        self.footprint       = set(state.footprint)
        self.energy          = state.energy
        self.on_boundary     = state.on_boundary
        self.processing_time = state.processing_time
        return self

    def copy(self):
        return Candidate().set(self)
    
    def fill_foreground(self, out, value=True):
        sel = np.s_[self.fg_offset[0] : self.fg_offset[0] + self.fg_fragment.shape[0], self.fg_offset[1] : self.fg_offset[1] + self.fg_fragment.shape[1]]
        out[sel] = value * self.fg_fragment
        return sel


def _get_economic_mask(y, mask, min_background_margin, max_background_margin):
    img_boundary_mask = np.zeros(mask.shape, bool)
    img_boundary_mask[ 0,  :] = True
    img_boundary_mask[-1,  :] = True
    img_boundary_mask[ :,  0] = True
    img_boundary_mask[ :, -1] = True
    mask_foreground  = np.logical_and(y.model > 0, mask)
    image_foreground = np.logical_and(y.model > 0, y.mask)
    tmp01 = (ndi.distance_transform_edt(~image_foreground) <= min_background_margin)
    image_background_within_margin = np.logical_and(tmp01, y.model < 0)
    foreground_clusters = ndi.label(image_foreground)[0]
    current_cluster_foreground = np.zeros(mask.shape, bool)
    for l in frozenset(foreground_clusters.reshape(-1)) - {0}:
        cc = (foreground_clusters == l)
        if mask[cc].any():
            current_cluster_foreground[cc] = True
    image_background_within_margin = np.logical_and(image_background_within_margin, ndi.distance_transform_edt(~current_cluster_foreground) <= max_background_margin)
    current_cluster_foreground = morph.binary_dilation(current_cluster_foreground, morph.disk(1))
    tmp02 = ndi.label(image_background_within_margin)[0]
    for l in frozenset(tmp02.reshape(-1)) - {0}:
        cc = (tmp02 == l)
        if current_cluster_foreground[cc].any():
            bg_mask = cc
            break
    economical_mask = np.logical_or(bg_mask, mask_foreground)
    if img_boundary_mask[current_cluster_foreground].any():
        economical_mask = np.logical_or(np.logical_and(img_boundary_mask, y.model < 0), economical_mask)
    return economical_mask


def _get_modelfit_region(candidate, y, g_atoms, min_background_margin, max_background_margin):
    region = candidate.get_region(y, g_atoms)
    region.mask = _get_economic_mask(y, region.mask, min_background_margin, max_background_margin)
    return region


def _process_candidate(y, g_atoms, x_map, candidate, modelfit_kwargs, smooth_mat_allocation_lock):
    modelfit_kwargs = gocell.aux.copy_dict(modelfit_kwargs)
    min_background_margin = max((modelfit_kwargs.pop('min_background_margin'), modelfit_kwargs['smooth_subsample']))
    max_background_margin =      modelfit_kwargs.pop('max_background_margin')
    region = _get_modelfit_region(candidate, y, g_atoms, min_background_margin, max_background_margin)
    for infoline in ('y.mask.sum()', 'region.mask.sum()', 'modelfit_kwargs'):
        print(f'{infoline}: {eval(infoline)}')
    t0 = time.time()
    J, result, fallback = _modelfit(region, smooth_mat_allocation_lock=smooth_mat_allocation_lock, **modelfit_kwargs)
    dt = time.time() - t0
    padded_mask = np.pad(region.mask, 1)
    smooth_mat  = gocell.aux.uplift_smooth_matrix(J.smooth_mat, padded_mask)
    padded_foreground = (result.map_to_image_pixels(y, region, pad=1).s(x_map, smooth_mat) > 0)
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
    candidate.processing_time = dt
    return candidate, fallback


@ray.remote
def _process_candidate_logged(log_root_dir, cidx, *args, **kwargs):
    if log_root_dir is not None:
        log_filename = gocell.aux.join_path(log_root_dir, f'{cidx}.txt')
        with io.TextIOWrapper(open(log_filename, 'wb', 0), write_through=True) as log_file:
            with contextlib.redirect_stdout(log_file):
                try:
                    result = _process_candidate(*args, **kwargs)
                except:
                    traceback.print_exc(file=log_file)
                    raise
    else:
        with contextlib.redirect_stdout(None):
            result = _process_candidate(*args, **kwargs)
    return (cidx, *result)


def process_candidates(candidates, y, g_atoms, modelfit_kwargs, log_root_dir, out=None):
    out = gocell.aux.get_output(out)
    modelfit_kwargs = gocell.aux.copy_dict(modelfit_kwargs)
    smooth_mat_max_allocations = modelfit_kwargs.pop('smooth_mat_max_allocations', np.inf)
    with gocell.aux.SystemSemaphore('smooth-matrix-allocation', smooth_mat_max_allocations) as smooth_mat_allocation_lock:
        candidates   = list(candidates)
        y_id         = ray.put(y)
        g_atoms_id   = ray.put(g_atoms)
        x_map_id     = ray.put(y.get_map(normalized=False, pad=1))
        mf_kwargs_id = ray.put(modelfit_kwargs)
        lock_id      = ray.put(smooth_mat_allocation_lock)
        futures      = [_process_candidate_logged.remote(log_root_dir, cidx, y_id, g_atoms_id, x_map_id, c, mf_kwargs_id, lock_id) for cidx, c in enumerate(candidates)]
        fallbacks    = 0
        for ret_idx, ret in enumerate(gocell.aux.get_ray_1by1(futures)):
            candidates[ret[0]].set(ret[1])
            out.intermediate(f'Processing candidates... {ret_idx + 1} / {len(futures)} ({fallbacks}x fallback)')
            if ret[2]: fallbacks += 1
    out.write(f'Processed candidates: {len(candidates)} ({fallbacks}x fallback)')


def _estimate_initialization(region):
    fg = region.model.copy()
    fg[~region.mask] = 0
    fg = (fg > 0)
    roi_xmap = region.get_map()
    fg_center = np.round(ndi.center_of_mass(fg)).astype(int)##
    fg_center = roi_xmap[:, fg_center[0], fg_center[1]]
    halfaxes_lengths = (roi_xmap[:, fg] - fg_center[:, None]).std(axis=1)
    return gocell.modelfit.PolynomialModel.create_ellipsoid(np.empty(0), fg_center, *halfaxes_lengths, np.eye(2))


def _print_cvxopt_solution(solution):
    print({key: solution[key] for key in ('status', 'gap', 'relative gap', 'primal objective', 'dual objective', 'primal slack', 'dual slack', 'primal infeasibility', 'dual infeasibility')})


def _fmt_timestamp(): return time.strftime('%X')


def _print_heading(line): print(f'-- {_fmt_timestamp()} -- {line} --')


def _modelfit(region, scale, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, smooth_mat_allocation_lock, smooth_mat_dtype, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None):
    _print_heading('initializing')
    smooth_matrix_factory = gocell.modelfit.SmoothMatrixFactory(smooth_amount, gaussian_shape_multiplier, smooth_subsample, smooth_mat_allocation_lock, smooth_mat_dtype)
    J = gocell.modelfit.Energy(region, epsilon, rho, smooth_matrix_factory, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest, 'scale': scale / J.smooth_mat.shape[0]}
    print(f'scale: {CP_params["scale"]:g}')
    fallback = False
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'gocell':
            _print_heading('convex programming starting: GOCELL')
            J_gocell = gocell.modelfit.Energy(region, epsilon, rho, gocell.modelfit.SmoothMatrixFactory.NULL_FACTORY)
            params = _estimate_initialization(region).array
            for retry in [False, True]:
                if not retry:
                    params = np.zeros(6)
                else:
                    print(f'-- retry --')
                    params = _estimate_initialization(region)
                    params_value = J_gocell(params)
                    print(f'initialization: {params_value}')
                    params = params.array
                    if params_value > J_gocell(solution):
                        print('initialization worse than previous solution - skipping retry')
                        break
                solution_info = gocell.modelfit.CP(J_gocell, params, **CP_params).solve()
                solution, status = np.array(solution_info['x']), solution_info['status']
                _print_cvxopt_solution(solution_info)
                if status == 'optimal': break
            params = gocell.modelfit.PolynomialModel(np.array(solution)).array
            print(f'solution: {J_gocell(params)}')
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    try:
        _print_heading('convex programming starting: GOCELLOS')
        solution_info = gocell.modelfit.CP(J, params, **CP_params).solve()
        solution, status = np.array(solution_info['x']), solution_info['status']
        _print_cvxopt_solution(solution_info)
        if status == 'unknown' and J(solution) > J(params):
            fallback = True ## numerical difficulties lead to a very bad solution, thus fall back to the GOCELL solution
        else:
            print(f'solution: {J(solution)}')
    except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc(file=sys.stdout)
        fallback = True  ## at least something we can continue the work with
    if fallback:
        _print_heading('GOCELLOS failed: falling back to GOCELL result')
        solution = params
    _print_heading('finished')
    return J, gocell.modelfit.PolynomialModel(solution), fallback


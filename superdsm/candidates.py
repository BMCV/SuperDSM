from ._aux import copy_dict, uplift_smooth_matrix, join_path, SystemSemaphore, get_ray_1by1
from .output import get_output
from .modelfit import PolynomialModel, CP, SmoothMatrixFactory, Energy

import ray
import sys, io, contextlib, traceback, time
import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


class BaseCandidate:
    def __init__(self):
        self.fg_offset   = None
        self.fg_fragment = None
    
    def fill_foreground(self, out, value=True):
        sel = np.s_[self.fg_offset[0] : self.fg_offset[0] + self.fg_fragment.shape[0], self.fg_offset[1] : self.fg_offset[1] + self.fg_fragment.shape[1]]
        out[sel] = value * self.fg_fragment
        return sel


class Candidate(BaseCandidate):
    def __init__(self):
        self.footprint       = set()
        self.energy          = np.nan
        self.on_boundary     = np.nan
        self.is_optimal      = np.nan
        self.processing_time = np.nan
        self._default_kwargs = {}

    def _update_default_kwarg(self, kwarg_name, value):
        if value is None:
            if kwarg_name in self._default_kwargs:
                return self._default_kwargs[kwarg_name]
            else:
                raise ValueError(f'kwarg "{kwarg_name}" not set yet')
        elif kwarg_name not in self._default_kwargs:
            self._default_kwargs[kwarg_name] = value
        return value
    
    def get_mask(self, g_atoms):
        return np.in1d(g_atoms, list(self.footprint)).reshape(g_atoms.shape)

    def get_modelfit_region(self, y, g_atoms, min_background_margin=None):
        min_background_margin = self._update_default_kwarg('min_background_margin', min_background_margin)
        region = y.get_region(self.get_mask(g_atoms))
        region.mask = np.logical_and(region.mask, ndi.distance_transform_edt(y.model <= 0) <= min_background_margin)
        return region

    def set(self, state):
        self.fg_fragment     = state.fg_fragment.copy() if state.fg_fragment is not None else None
        self.fg_offset       = state.fg_offset.copy() if state.fg_offset is not None else None
        self.footprint       = set(state.footprint)
        self.energy          = state.energy
        self.on_boundary     = state.on_boundary
        self.is_optimal      = state.is_optimal
        self.processing_time = state.processing_time
        self._default_kwargs = copy_dict(state._default_kwargs)
        return self

    def copy(self):
        return Candidate().set(self)


def extract_foreground_fragment(fg_mask):
    if fg_mask.any():
        rows = fg_mask.any(axis=1)
        cols = fg_mask.any(axis=0)
        rmin, rmax  = np.where(rows)[0][[0, -1]]
        cmin, cmax  = np.where(cols)[0][[0, -1]]
        fg_offset   = np.array([rmin, cmin])
        fg_fragment = fg_mask[rmin : rmax + 1, cmin : cmax + 1]
        return fg_offset, fg_fragment
    else:
        return np.zeros(2, int), np.zeros((1, 1), bool)


def _process_candidate(y, g_atoms, x_map, candidate, modelfit_kwargs, smooth_mat_allocation_lock):
    modelfit_kwargs = copy_dict(modelfit_kwargs)
    min_background_margin = max((modelfit_kwargs.pop('min_background_margin'), modelfit_kwargs['smooth_subsample']))
    region = candidate.get_modelfit_region(y, g_atoms, min_background_margin)
    for infoline in ('y.mask.sum()', 'region.mask.sum()', 'np.logical_and(region.model > 0, region.mask).sum()', 'modelfit_kwargs'):
        print(f'{infoline}: {eval(infoline)}')

    # Skip regions whose foreground is only a single pixel (this is just noise)
    if (region.model[region.mask] > 0).sum() == 1:
        candidate.fg_offset   = np.zeros(2, int)
        candidate.fg_fragment = np.zeros((1, 1), bool)
        candidate.energy      = 0.
        candidate.on_boundary = False
        candidate.is_optimal  = False
        candidate.processing_time = 0
        return candidate, False

    # Otherwise, perform model fitting
    else:
        t0 = time.time()
        J, result, status = _modelfit(region, smooth_mat_allocation_lock=smooth_mat_allocation_lock, **modelfit_kwargs)
        dt = time.time() - t0
        padded_mask = np.pad(region.mask, 1)
        smooth_mat  = uplift_smooth_matrix(J.smooth_mat, padded_mask)
        padded_foreground = (result.map_to_image_pixels(y, region, pad=1).s(x_map, smooth_mat) > 0)
        foreground = padded_foreground[1:-1, 1:-1]
        if foreground.any():
            foreground = np.logical_and(region.mask, foreground)
            candidate.fg_offset, candidate.fg_fragment = extract_foreground_fragment(foreground)
        else:
            candidate.fg_offset   = np.zeros(2, int)
            candidate.fg_fragment = np.zeros((1, 1), bool)
        candidate.energy      = J(result)
        candidate.on_boundary = padded_foreground[0].any() or padded_foreground[-1].any() or padded_foreground[:, 0].any() or padded_foreground[:, -1].any()
        candidate.is_optimal  = (status == 'optimal')
        candidate.processing_time = dt
        return candidate, (status == 'fallback')


@ray.remote
def _ray_process_candidate_logged(*args, **kwargs):
    return _process_candidate_logged(*args, **kwargs)


def _process_candidate_logged(log_root_dir, cidx, *args, **kwargs):
    try:
        if log_root_dir is not None:
            log_filename = join_path(log_root_dir, f'{cidx}.txt')
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
    except ModelfitError as error:
        error.cidx = cidx
        raise
    return (cidx, *result)


DEFAULT_PROCESSING_STATUS_LINE = ('Processing candidates', 'Processed candidates')


def process_candidates(candidates, y, g_atoms, modelfit_kwargs, log_root_dir, status_line=DEFAULT_PROCESSING_STATUS_LINE, out=None):
    out = get_output(out)
    modelfit_kwargs = copy_dict(modelfit_kwargs)
    smooth_mat_max_allocations = modelfit_kwargs.pop('smooth_mat_max_allocations', np.inf)
    with SystemSemaphore('smooth-matrix-allocation', smooth_mat_max_allocations) as smooth_mat_allocation_lock:
        candidates = list(candidates)
        fallbacks  = 0
        x_map      = y.get_map(normalized=False, pad=1)
        for ret_idx, ret in enumerate(_process_candidates(candidates, y, g_atoms, x_map, smooth_mat_allocation_lock, modelfit_kwargs, log_root_dir)):
            candidates[ret[0]].set(ret[1])
            out.intermediate(f'{status_line[0]}... {ret_idx + 1} / {len(candidates)} ({fallbacks}x fallback)')
            if ret[2]: fallbacks += 1
    out.write(f'{status_line[1]}: {len(candidates)} ({fallbacks}x fallback)')


def _process_candidates(candidates, y, g_atoms, x_map, lock, modelfit_kwargs, log_root_dir):
    if _process_candidates._DEBUG: ## run serially
        for cidx, c in enumerate(candidates):
            yield _process_candidate_logged(log_root_dir, cidx, y, g_atoms, x_map, c, modelfit_kwargs, lock)
    else: ## run in parallel
        y_id         = ray.put(y)
        g_atoms_id   = ray.put(g_atoms)
        x_map_id     = ray.put(x_map)
        mf_kwargs_id = ray.put(modelfit_kwargs)
        lock_id      = ray.put(lock)
        futures      = [_ray_process_candidate_logged.remote(log_root_dir, cidx, y_id, g_atoms_id, x_map_id, c, mf_kwargs_id, lock_id) for cidx, c in enumerate(candidates)]
        for ret in get_ray_1by1(futures): yield ret


_process_candidates._DEBUG = False


def _estimate_initialization(region):
    fg = region.model.copy()
    fg[~region.mask] = 0
    fg = (fg > 0)
    roi_xmap = region.get_map()
    fg_center = np.round(ndi.center_of_mass(fg)).astype(int)
    fg_center = roi_xmap[:, fg_center[0], fg_center[1]]
    halfaxes_lengths = (roi_xmap[:, fg] - fg_center[:, None]).std(axis=1)
    halfaxes_lengths = np.max([halfaxes_lengths, np.full(halfaxes_lengths.shape, 1e-8)], axis=0)
    return PolynomialModel.create_ellipsoid(np.empty(0), fg_center, *halfaxes_lengths, np.eye(2))


def _print_cvxopt_solution(solution):
    print({key: solution[key] for key in ('status', 'gap', 'relative gap', 'primal objective', 'dual objective', 'primal slack', 'dual slack', 'primal infeasibility', 'dual infeasibility')})


def _fmt_timestamp(): return time.strftime('%X')


def _print_heading(line): print(f'-- {_fmt_timestamp()} -- {line} --')


class ModelfitError(Exception):
    def __init__(self, *args, cidx=None, cause=None):
        super().__init__(*args)
        self.cidx = cidx

    def __str__(self):
        messages = [str(arg) for arg in self.args]
        if self.cidx is not None:
            messages.append(f'cidx: {self.cidx}')
        return ', '.join(messages)


def _compute_elliptical_solution(J_elliptical, CP_params):
    solution_info  = None
    solution_array = None
    solution_value = np.inf

    # Pass 1: Try zeros initialization
    try:
        solution_info  = CP(J_elliptical, np.zeros(6), **CP_params).solve()
        solution_array = PolynomialModel(np.array(solution_info['x'])).array
        solution_value = J_elliptical(solution_array)
        print(f'solution: {solution_value}')
    except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc()
        pass ## continue with Pass 2 (retry)

    # Pass 2: Try data-specific initialization
    if solution_info is None or solution_info['status'] != 'optimal':
        print(f'-- retry --')
        initialization = _estimate_initialization(J_elliptical.roi)
        initialization_value = J_elliptical(initialization)
        print(f'initialization: {initialization_value}')
        if initialization_value > solution_value:
            print('initialization worse than previous solution - skipping retry')
        else:
            try:
                solution_info  = CP(J_elliptical, initialization.array, **CP_params).solve()
                solution_array = PolynomialModel(np.array(solution_info['x'])).array
                solution_value = J_elliptical(solution_array)
                print(f'solution: {solution_value}')
            except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
                if solution_info is None:
                    cause = sys.exc_info()[1]
                    raise ModelfitError(cause)
                else:
                    pass ## continue with previous solution (Pass 1)

    assert solution_array is not None
    return solution_array


def _modelfit(region, scale, epsilon, alpha, smooth_amount, smooth_subsample, gaussian_shape_multiplier, smooth_mat_allocation_lock, smooth_mat_dtype, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None, cp_timeout=None):
    _print_heading('initializing')
    smooth_matrix_factory = SmoothMatrixFactory(smooth_amount, gaussian_shape_multiplier, smooth_subsample, smooth_mat_allocation_lock, smooth_mat_dtype)
    J = Energy(region, epsilon, alpha, smooth_matrix_factory, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest, 'scale': scale / J.smooth_mat.shape[0], 'timeout': cp_timeout}
    print(f'scale: {CP_params["scale"]:g}')
    status = None
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'elliptical':
            _print_heading('convex programming starting: using elliptical models')
            J_elliptical = Energy(region, epsilon, alpha, SmoothMatrixFactory.NULL_FACTORY)
            params = _compute_elliptical_solution(J_elliptical, CP_params)
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    try:
        _print_heading('convex programming starting: using deformable shape models (DSM)')
        solution_info = CP(J, params, **CP_params).solve()
        solution = np.array(solution_info['x'])
        _print_cvxopt_solution(solution_info)
        if solution_info['status'] == 'unknown' and J(solution) > J(params):
            status = 'fallback' ## numerical difficulties lead to a very bad solution, thus fall back to the elliptical solution
        else:
            print(f'solution: {J(solution)}')
            status = 'optimal'
    except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc(file=sys.stdout)
        status = 'fallback'  ## at least something we can continue the work with
    assert status is not None
    if status == 'fallback':
        _print_heading('DSM failed: falling back to elliptical result')
        solution = params
    _print_heading('finished')
    return J, PolynomialModel(solution), status


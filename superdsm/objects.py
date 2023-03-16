from ._aux import copy_dict, uplift_smooth_matrix, join_path, SystemSemaphore, get_ray_1by1
from .output import get_output
from .dsm import DeformableShapeModel, CP, SmoothMatrixFactory, Energy

import ray
import sys, io, contextlib, traceback, time
import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


class BaseObject:
    """Each object of this class represents a segmentation mask, consisting of a *foreground fragment* and an *offset*.
    """

    def __init__(self):
        self.fg_offset   = None
        self.fg_fragment = None
    
    def fill_foreground(self, out, value=True):
        """Reproduces the segmentation mask of this object.
        
        The foreground fragment is written into the image ``out``, which must be an object of ``numpy.ndarray`` type. Image points corresponding to the segmentation mask will be set to ``value``.

        :return: The slice corresponding to the altered region of ``out``.

        .. runblock:: pycon

           >>> import superdsm.objects
           >>> import numpy as np
           >>> obj = superdsm.objects.BaseObject()
           >>> obj.fg_fragment = np.array([[False,  True],
           ...                             [ True,  True],
           ...                             [ True, False]])
           >>> obj.fg_offset = (1, 2)
           >>> mask = np.zeros((4, 5), bool)
           >>> obj.fill_foreground(mask)
           >>> mask
        
        This method is the counterpart of the :py:meth:`~.extract_foreground_fragment` function.
        """
        sel = np.s_[self.fg_offset[0] : self.fg_offset[0] + self.fg_fragment.shape[0], self.fg_offset[1] : self.fg_offset[1] + self.fg_fragment.shape[1]]
        out[sel] = value * self.fg_fragment
        return sel


class Object(BaseObject):
    """Each object of this class represents a set of atomic image regions.

    Each object corresponds to a realization of the set :math:`X` in the paper (see :ref:`Section 3 <references>`). It also represents a segmented object after it has been passed to the :py:meth:`compute_objects` function.
    """

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
    
    def get_mask(self, atoms):
        """Returns binary image corresponding to the union of the represented set of atomic image regions.

        :param atoms: Integer-valued image representing the universe of atomic image regions (each atomic image region has a unique label, which is the integer value).
        :return: Binary image corresponding to :math:`\\tilde\\omega(X) = \\bigcup X` in the paper, where each object of this class corresponds to a realization of the set :math:`X` (see :ref:`Section 3 <references>`).

        .. runblock:: pycon

           >>> import superdsm.objects
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2],
           ...                   [1, 3, 2],
           ...                   [3, 3, 3]])
           >>> obj = superdsm.objects.Object()
           >>> obj.footprint = set([2, 3])
           >>> obj.get_mask()
        """
        return np.in1d(atoms, list(self.footprint)).reshape(atoms.shape)

    def get_cvxprog_region(self, y, atoms, min_background_margin=None):
        min_background_margin = self._update_default_kwarg('min_background_margin', min_background_margin)
        region = y.get_region(self.get_mask(atoms))
        region.mask = np.logical_and(region.mask, ndi.distance_transform_edt(y.model <= 0) <= min_background_margin)
        return region

    def set(self, state):
        """Adopts the state of another object.
        """
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
        """Returns a deep copy of this object.
        """
        return Object().set(self)


def extract_foreground_fragment(fg_mask):
    """Returns the minimal-size rectangle region of image foreground and the corresponding offset.

    .. runblock:: pycon

       >>> import superdsm.objects
       >>> import numpy as np
       >>> mask = np.array([[False, False, False, False, False],
       ...                  [False, False, False,  True, False],
       ...                  [False, False,  True,  True, False],
       ...                  [False, False,  True, False, False]])
       >>> offset, fragment = superdsm.objects.extract_foreground_fragment(mask)
       >>> offset
       >>> fragment
    
    This function is the counterpart of the :py:meth:`~.BaseObject.fill_foreground` method.
    """
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


def _compute_object(y, atoms, x_map, object, cvxprog_kwargs, smooth_mat_allocation_lock):
    cvxprog_kwargs = copy_dict(cvxprog_kwargs)
    min_background_margin = max((cvxprog_kwargs.pop('min_background_margin'), cvxprog_kwargs['smooth_subsample']))
    region = object.get_cvxprog_region(y, atoms, min_background_margin)
    for infoline in ('y.mask.sum()', 'region.mask.sum()', 'np.logical_and(region.model > 0, region.mask).sum()', 'cvxprog_kwargs'):
        print(f'{infoline}: {eval(infoline)}')

    # Skip regions whose foreground is only a single pixel (this is just noise)
    if (region.model[region.mask] > 0).sum() == 1:
        object.fg_offset   = np.zeros(2, int)
        object.fg_fragment = np.zeros((1, 1), bool)
        object.energy      = 0.
        object.on_boundary = False
        object.is_optimal  = False
        object.processing_time = 0
        return object, False

    # Otherwise, perform model fitting
    else:
        t0 = time.time()
        J, result, status = cvxprog(region, smooth_mat_allocation_lock=smooth_mat_allocation_lock, **cvxprog_kwargs)
        dt = time.time() - t0
        padded_mask = np.pad(region.mask, 1)
        smooth_mat  = uplift_smooth_matrix(J.smooth_mat, padded_mask)
        padded_foreground = (result.map_to_image_pixels(y, region, pad=1).s(x_map, smooth_mat) > 0)
        foreground = padded_foreground[1:-1, 1:-1]
        if foreground.any():
            foreground = np.logical_and(region.mask, foreground)
            object.fg_offset, object.fg_fragment = extract_foreground_fragment(foreground)
        else:
            object.fg_offset   = np.zeros(2, int)
            object.fg_fragment = np.zeros((1, 1), bool)
        object.energy      = J(result)
        object.on_boundary = padded_foreground[0].any() or padded_foreground[-1].any() or padded_foreground[:, 0].any() or padded_foreground[:, -1].any()
        object.is_optimal  = (status == 'optimal')
        object.processing_time = dt
        return object, (status == 'fallback')


@ray.remote
def _ray_compute_object_logged(*args, **kwargs):
    return _compute_object_logged(*args, **kwargs)


def _compute_object_logged(log_root_dir, cidx, *args, **kwargs):
    try:
        if log_root_dir is not None:
            log_filename = join_path(log_root_dir, f'{cidx}.txt')
            with io.TextIOWrapper(open(log_filename, 'wb', 0), write_through=True) as log_file:
                with contextlib.redirect_stdout(log_file):
                    try:
                        result = _compute_object(*args, **kwargs)
                    except:
                        traceback.print_exc(file=log_file)
                        raise
        else:
            with contextlib.redirect_stdout(None):
                result = _compute_object(*args, **kwargs)
    except CvxprogError as error:
        error.cidx = cidx
        raise
    return (cidx, *result)


DEFAULT_COMPUTING_STATUS_LINE = ('Computing objects', 'Computed objects')


def compute_objects(objects, y, atoms, cvxprog_kwargs, log_root_dir, status_line=DEFAULT_COMPUTING_STATUS_LINE, out=None):
    out = get_output(out)
    cvxprog_kwargs = copy_dict(cvxprog_kwargs)
    smooth_mat_max_allocations = cvxprog_kwargs.pop('smooth_mat_max_allocations', np.inf)
    with SystemSemaphore('smooth-matrix-allocation', smooth_mat_max_allocations) as smooth_mat_allocation_lock:
        objects = list(objects)
        fallbacks  = 0
        x_map      = y.get_map(normalized=False, pad=1)
        for ret_idx, ret in enumerate(_compute_objects(objects, y, atoms, x_map, smooth_mat_allocation_lock, cvxprog_kwargs, log_root_dir)):
            objects[ret[0]].set(ret[1])
            out.intermediate(f'{status_line[0]}... {ret_idx + 1} / {len(objects)} ({fallbacks}x fallback)')
            if ret[2]: fallbacks += 1
    out.write(f'{status_line[1]}: {len(objects)} ({fallbacks}x fallback)')


def _compute_objects(objects, y, atoms, x_map, lock, cvxprog_kwargs, log_root_dir):
    if _compute_objects._DEBUG: ## run serially
        for cidx, c in enumerate(objects):
            yield _compute_object_logged(log_root_dir, cidx, y, atoms, x_map, c, cvxprog_kwargs, lock)
    else: ## run in parallel
        y_id         = ray.put(y)
        atoms_id     = ray.put(atoms)
        x_map_id     = ray.put(x_map)
        cp_kwargs_id = ray.put(cvxprog_kwargs)
        lock_id      = ray.put(lock)
        futures      = [_ray_compute_object_logged.remote(log_root_dir, obj_idx, y_id, atoms_id, x_map_id, obj, cp_kwargs_id, lock_id) for obj_idx, obj in enumerate(objects)]
        for ret in get_ray_1by1(futures): yield ret


_compute_objects._DEBUG = False


def _estimate_initialization(region):
    fg = region.model.copy()
    fg[~region.mask] = 0
    fg = (fg > 0)
    roi_xmap = region.get_map()
    fg_center = np.round(ndi.center_of_mass(fg)).astype(int)
    fg_center = roi_xmap[:, fg_center[0], fg_center[1]]
    halfaxes_lengths = (roi_xmap[:, fg] - fg_center[:, None]).std(axis=1)
    halfaxes_lengths = np.max([halfaxes_lengths, np.full(halfaxes_lengths.shape, 1e-8)], axis=0)
    return DeformableShapeModel.create_ellipsoid(np.empty(0), fg_center, *halfaxes_lengths, np.eye(2))


def _print_cvxopt_solution(solution):
    print({key: solution[key] for key in ('status', 'gap', 'relative gap', 'primal objective', 'dual objective', 'primal slack', 'dual slack', 'primal infeasibility', 'dual infeasibility')})


def _fmt_timestamp(): return time.strftime('%X')


def _print_heading(line): print(f'-- {_fmt_timestamp()} -- {line} --')


class CvxprogError(Exception):
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
        solution_array = DeformableShapeModel(np.array(solution_info['x'])).array
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
                solution_array = DeformableShapeModel(np.array(solution_info['x'])).array
                solution_value = J_elliptical(solution_array)
                print(f'solution: {solution_value}')
            except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
                if solution_info is None:
                    cause = sys.exc_info()[1]
                    raise CvxprogError(cause)
                else:
                    pass ## continue with previous solution (Pass 1)

    assert solution_array is not None
    return solution_array


def cvxprog(region, scale, epsilon, alpha, smooth_amount, smooth_subsample, gaussian_shape_multiplier, smooth_mat_allocation_lock, smooth_mat_dtype, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None, cp_timeout=None):
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
    return J, DeformableShapeModel(solution), status


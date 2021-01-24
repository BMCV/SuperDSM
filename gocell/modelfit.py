import gocell.surface as surface
import gocell.aux     as aux
import gocell._mkl    as mkl

import numpy as np
import cvxopt
import skimage.util
import signal

from math         import sqrt
from scipy.linalg import orth
from scipy        import ndimage
from scipy.sparse import csr_matrix, coo_matrix, bmat as sparse_block, diags as sparse_diag, issparse


def mkl_dot(A, B):
    if A.shape[1] == B.shape[0] == 1: return A @ B
    return mkl.dot(A, B)


class PolynomialModelType:
    
    def get_model(self, params):
        model = params if isinstance(params, PolynomialModel) else PolynomialModel(params)
        assert not np.isnan(model.array).any()
        return model
    
    def compute_derivatives(self, x_map):
        derivatives = [None] * 6
        derivatives[0] = np.square(x_map[0])
        derivatives[1] = np.square(x_map[1])
        derivatives[2] = 2 * np.prod([x_map[i] for i in range(2)], axis=0)
        derivatives[3] = 2 * x_map[0]
        derivatives[4] = 2 * x_map[1]
        derivatives[5] = 1
        return derivatives


class PolynomialModel:
    
    TYPE = PolynomialModelType()
    
    def __init__(self, *args):
        if len(args) == 1 and len(args[0]) >= 6:
            self.array = args[0].astype(float).reshape(-1)
            self.a = args[0].flat[:3    ]
            self.b = args[0].flat[ 3:5  ]
            self.c = args[0].flat[   5  ]
            self.ξ = args[0].flat[    6:]
        elif len(args) >= 1:
            assert isinstance(args[0], (int, np.ndarray))
            self.ξ = np.zeros(args[0])   if isinstance(args[0], int) else args[0].reshape(-1)
            self.a = np.array([1, 1, 0]) if len(args) < 2 else args[1].flat[np.array([0, 3, 1])]
            self.b =         np.zeros(2) if len(args) < 3 else args[2].astype(float)
            self.c =                   0 if len(args) < 4 else float(args[3])
            self.array = np.concatenate([self.a, self.b, np.array([self.c]), self.ξ])
        else:
            raise ValueError('Initialization failed')

    def copy(self):
        return PolynomialModel(self.array.copy())
    
    @property
    def A(self):
        return np.array([self.a[0], self.a[2], self.a[2], self.a[1]]).reshape((2, 2))
    
    def s(self, x, smooth_mat):
        xdim = x.ndim - 1 if isinstance(x, np.ndarray) else 0
        xvec = np.array(x).reshape((2, -1))
        svec = diagquad(self.A, xvec) + 2 * np.inner(xvec.T, self.b) + self.c + mkl.dot(smooth_mat, self.ξ)
        return svec.reshape(x.shape[-xdim:]) if isinstance(x, np.ndarray) else svec
    
    @staticmethod
    def create_ellipsoid(ξ, center, halfaxis1_len, halfaxis2_len, U=None):
        ev = lambda half_length: (1. / np.square(half_length))
        if U is None: U = orth(np.random.randn(2, 2)) # random rotation matrix
        A  = U.dot(np.diag((ev(halfaxis1_len), ev(halfaxis2_len)))).dot(U.T)
        b  = A.dot(center)
        c  = np.inner(center, b) - 1
        return PolynomialModel(ξ, -A, b, -c)

    def is_measurable(self):
        return (np.linalg.eigvalsh(self.A) < 0).all()
    
    def measure(self):
        """Returns the center `b` and halfaxes `H` of an ellipse-shaped model.
        
        The matrix `H` consists of two rows, which correspond to the two
        halfaxes. Thus, the two halfaxes can be accessed as `H[0]` and `H[1]`.
        """
        b = -np.linalg.inv(self.A + 1e-12 * np.eye(self.A.ndim)).dot(self.b)
        c = self.c - np.inner(b, self.A.dot(b))
        d,U = np.linalg.eigh(self.A)
        l = np.sqrt(np.max((np.zeros_like(d), -c / d), axis=0))
        L = U.dot(np.diag(l))
        return b, L.T
    
    def map_to_image_pixels(self, g, roi, pad=0):
        assert pad >= 0 and isinstance(pad, int)
        g_max_coord, roi_max_coord = 2 * pad + np.array(g.model.shape) - 1., np.array(roi.model.shape) - 1.
        G = np.diag(1. / roi_max_coord)
        v = -G.dot(np.add(roi.offset, pad))
        A = G.dot(self.A).dot(G)
        b = G.dot(self.A.dot(v) + self.b)
        c = np.inner(v, self.A.dot(v)) + 2 * np.inner(self.b, v) + self.c
        return PolynomialModel(self.ξ, A, b, c)


def diagquad(A, X):
    """Computes the diagonal entries of `X' A X` quickly.
    
    See: http://stackoverflow.com/a/14759341/1444073
    """
    return np.einsum('ij,ij->i', np.dot(X.T, A), X.T)


def _create_gaussian_kernel(sigma, shape=None, shape_multiplier=1):
    if abs(shape_multiplier - 1) > 0 and shape is not None: raise ValueError()
    if shape is None: shape = [round(1 + sigma * 4 * shape_multiplier)] * 2
    inp = np.zeros(shape)
    inp[shape[0] // 2, shape[1] // 2] = 1
    return ndimage.gaussian_filter(inp, sigma)


def _convmat(filter_mask, img_shape, row_mask=None, col_mask=None, lock=None):
    assert filter_mask.ndim == 2 and filter_mask.shape[0] == filter_mask.shape[1]
    assert filter_mask.shape[0] % 2 == 1
    if row_mask is None: row_mask = np.ones(img_shape, bool)
    if col_mask is None: col_mask = np.ones(img_shape, bool)
    print('.', end='')
    p = np.subtract(img_shape, filter_mask.shape[0] // 2 + 1)
    assert (p >= 0).all(), f'filter_mask {filter_mask.shape} too large for img_shape {img_shape}'
    print('.', end='')
    z = np.pad(filter_mask, np.vstack([p, p]).T)
    print('.', end='')
    z = skimage.util.view_as_windows(z, img_shape)[::-1, ::-1]
    print('.', end='\n')
    with aux.SystemSemaphore.get_lock(lock):
        col_mask_where = np.nonzero(col_mask)
        row_mask_where = np.nonzero(row_mask)
        return z[row_mask_where[0][:,None], row_mask_where[1][:,None], col_mask_where[0], col_mask_where[1]]


def _create_subsample_grid(mask, subsample, mask_offset=(0,0)):
    grid_offset = np.asarray(mask_offset) % subsample
    subsample_grid = np.zeros_like(mask)
    subsample_grid[grid_offset[0]::subsample, grid_offset[1]::subsample] = True
    subsample_grid = np.logical_and(mask, subsample_grid)
    distances = mask * ndimage.distance_transform_bf(~subsample_grid, metric='chessboard')
    tmp1 = np.ones_like(subsample_grid, bool)
    while True:
        outside = (distances >= subsample)
        if not outside.any(): break
        min_outside_distance = distances[outside].min()
        min_outside_pixel = tuple(np.asarray(np.where(distances == min_outside_distance)).T[0])
        subsample_grid[min_outside_pixel] = True
        tmp1[min_outside_pixel] = False
        tmp2 = ndimage.distance_transform_bf(tmp1, metric='chessboard')
        distances = np.min((distances, tmp2), axis=0)
        tmp1[min_outside_pixel] = True
    return subsample_grid


def _create_masked_smooth_matrix(kernel, mask, subsample=1, lock=None):
#    mask_offset = (np.where(mask.any(axis=1))[0][0], np.where(mask.any(axis=0))[0][0])
    mask = mask[np.where(mask.any(axis=1))[0], :]
    mask = mask[:, np.where(mask.any(axis=0))[0]]
    if (mask.shape <= np.asarray(kernel.shape) // 2).any(): return None
#    subsample_grid = _create_subsample_grid(mask, subsample, mask_offset)
    subsample_grid = _create_subsample_grid(mask, subsample)
    col_mask = np.logical_and(mask, subsample_grid)
    print(f'{mask.sum()} rows, {col_mask.sum()} columns')
    M = _convmat(kernel, mask.shape, row_mask=mask, col_mask=col_mask, lock=lock)
    M_sums = M.sum(axis=1)
    M /= M_sums[:, None]
    assert (M.sum(axis=0) > 0).all() and (M.sum(axis=1) > 0).all()
    return M


class SmoothMatrixFactory:

    def __init__(self, smooth_amount, shape_multiplier, smooth_subsample, lock=None, dtype='float32'):
        self.smooth_amount    = smooth_amount
        self.shape_multiplier = shape_multiplier
        self.smooth_subsample = smooth_subsample
        self.lock             = lock
        self.dtype            = dtype

    def get(self, mask, uplift=False):
        print('-- smooth matrix computation starting --')
        mat = None
        if self.smooth_amount < np.inf:
            psf = _create_gaussian_kernel(self.smooth_amount, shape_multiplier=self.shape_multiplier).astype(self.dtype)
            mat = _create_masked_smooth_matrix(psf, mask, self.smooth_subsample, self.lock)
            # NOTE: `mat` will be `None` if `psf` is too large for `mask`
        if mat is None:
            print('using null-matrix')
            mat = np.empty((mask.sum(), 0))
        mat = csr_matrix(mat).astype(np.float64, copy=False)
        if uplift: mat = aux.uplift_smooth_matrix(mat, mask)
        print('-- smooth matrix finished --')
        return mat
    
SmoothMatrixFactory.NULL_FACTORY = SmoothMatrixFactory(np.inf, np.nan, np.nan)


class Energy:

    def __init__(self, roi, epsilon, rho, smooth_matrix_factory, sparsity_tol=0, hessian_sparsity_tol=0, model_type=PolynomialModel.TYPE):
        self.roi = roi
        self.p   = None

        self.smooth_mat = smooth_matrix_factory.get(roi.mask)

        self.x = self.roi.get_map()[:, roi.mask]
        self.w = np.ones(roi.mask.sum(), 'uint8')
        self.y = roi.model[roi.mask]

        assert epsilon > 0, 'epsilon must be strictly positive'
        self.epsilon = epsilon

        assert rho >= 0, 'rho must be positive'
        self.rho = rho

        assert sparsity_tol >= 0, 'sparsity_tol must be positive'
        self.sparsity_tol = sparsity_tol

        assert hessian_sparsity_tol >= 0, 'hessian_sparsity_tol must be positive'
        self.hessian_sparsity_tol = hessian_sparsity_tol

        # pre-compute common terms occuring in the computation of the derivatives
        self.model_type = model_type
        self.q = model_type.compute_derivatives(self.x)
    
    def update_maps(self, params):
        if self.p is not None and all(self.p.array == params.array): return
        s = params.s(self.x, self.smooth_mat)
        self.p     = params
        self.t     = self.y * s
        self.theta = None # invalidate
        
        valid_t_mask = (self.t >= -np.log(np.finfo(self.t.dtype).max))
        self.h = np.full(self.t.shape, np.nan)
        self.h[valid_t_mask] = np.exp(-self.t[valid_t_mask])

        if self.smooth_mat.shape[1] > 0:
            self.term3 = np.square(params.ξ)
            self.term2 = np.sqrt(self.term3 + self.epsilon)
    
    def update_theta(self):
        if self.theta is None:
            valid_h_mask = ~np.isnan(self.h)
            self.theta = np.ones_like(self.t)
            self.theta[valid_h_mask] = self.h[valid_h_mask] / (1 + self.h[valid_h_mask])
    
    def __call__(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        valid_h_mask = ~np.isnan(self.h)
        phi = np.zeros_like(self.t)
        phi[ valid_h_mask] = np.log(1 + self.h[valid_h_mask])
        phi[~valid_h_mask] = -self.t[~valid_h_mask]
        objective1 = np.inner(self.w.flat, phi.flat)
        if self.smooth_mat.shape[1] > 0:
            objective2  = self.rho * self.term2.sum()
            objective2 -= self.rho * sqrt(self.epsilon) * len(self.term2)
            if objective2 < 0:
                assert np.allclose(0, objective2)
                objective2 = 0
            else:
                assert objective2 >= 0
        else:
            objective2 = 0
        return objective1 + objective2
    
    def grad(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        term1 = -self.y * self.theta
        grad = np.asarray([term1 * q for q in self.q]) @ self.w
        term1[abs(term1) < self.sparsity_tol] = 0
        term1_sparse = coo_matrix(term1).transpose(copy=False)
        if self.smooth_mat.shape[1] > 0:
            grad2  = (self.w.reshape(-1)[None, :] @ self.smooth_mat.multiply(term1_sparse)).reshape(-1)
            grad2 += self.rho * (params.ξ / self.term2)
            grad   = np.concatenate([grad, grad2])
        return grad
    
    def hessian(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        gamma = self.theta - np.square(self.theta)
        gamma[gamma < self.sparsity_tol] = 0
        pixelmask = (gamma != 0)
        term4 = np.sqrt(gamma[pixelmask] * self.w[pixelmask])[None, :]
        D1 = np.asarray([-self.y * qi for qi in self.q])[:, pixelmask] * term4
        D2 = self.smooth_mat[pixelmask].multiply(-self.y[pixelmask, None]).T.multiply(term4).tocsr()
        if self.smooth_mat.shape[1] > 0:
            H = sparse_block([
                [D1 @ D1.T, csr_matrix((D1.shape[0], D2.shape[0]))],
                [mkl_dot(D2, D1.T), mkl.gram(D2).T if D2.shape[1] > 0 else csr_matrix((D2.shape[0], D2.shape[0]))]])
            g = self.rho * (1 / self.term2 - self.term3 / np.power(self.term2, 3))
            assert np.allclose(0, g[g < 0])
            g[g < 0] = 0
            H += sparse_diag(np.concatenate([np.zeros(6), g]))
            if self.hessian_sparsity_tol > 0:
                H = H.tocoo()
                H_mask = (np.abs(H.data) >= self.hessian_sparsity_tol)
                H_mask = np.logical_or(H_mask, H.row == H.col)
                H.data = H.data[H_mask]
                H.row  = H.row [H_mask]
                H.col  = H.col [H_mask]
        else:
            H = D1 @ D1.T
        return H


class Cache:

    def __init__(self, size, getter, equality=None):
        if equality is None: equality = np.array_equal
        elif isinstance(equality, str): equality = eval(equality)
        assert callable(equality)
        self.size     = size
        self.inputs   = []
        self.outputs  = []
        self.getter   = getter
        self.equality = equality

    def __call__(self, input):
        pos = -1
        for i in range(len(self.inputs))[::-1]:
            input2 = self.inputs[i]
            if self.equality(input, input2):
                pos = i
                input = input2
                break
        if pos > -1:
            output = self.outputs[pos]
            del self.inputs[pos], self.outputs[pos]
        else:
            output = self.getter(input)
        self.inputs .append(input)
        self.outputs.append(output)
        assert len(self.inputs) == len(self.outputs)
        if len(self.inputs) > self.size:
            del self.inputs[0], self.outputs[0]
        return output


class TimeoutError(Exception):
    pass


def _cp_timeout_handler(*args):
    raise TimeoutError()


class CP:

    CHECK_NUMBERS = True

    def __init__(self, energy, params0, scale=1, cachesize=0, cachetest=None, timeout=None):
        self.params0  = params0
        self.gradient = Cache(cachesize, lambda p: (scale * energy(p), cvxopt.matrix(scale * energy.grad(p)).T), equality=cachetest)
        self.hessian  = Cache(cachesize, lambda p:  scale * energy.hessian(p), equality=cachetest)
        self.timeout  = timeout
    
    def __call__(self, params=None, w=None):
        if params is None:
            return 0, cvxopt.matrix(self.params0)
        else:
            p = np.array(params).reshape(-1)
            l, Dl = self.gradient(p)
            if CP.CHECK_NUMBERS:
                Dl_array = np.array(Dl)
                assert not np.isnan(p).any() and not np.isinf(p).any()
                assert not np.isnan(Dl_array).any() and not np.isinf(Dl_array).any()
            if w is None:
                return l, Dl
            else:
                H = self.hessian(p)
                if issparse(H):
                    if CP.CHECK_NUMBERS:
                        H_array = H.toarray()
                        assert not np.isnan(H_array).any() and not np.isinf(H_array).any()
                    H = H.tocoo()
                    H = cvxopt.spmatrix(w[0] * H.data, H.row, H.col, size=H.shape)
                else:
                    if CP.CHECK_NUMBERS:
                        assert not np.isnan(H).any() and not np.isinf(H).any()
                    H = cvxopt.matrix(w[0] * H)
                return l, Dl, H
    
    def solve(self, **options):
        if self.timeout is not None and self.timeout > 0:
            signal.signal(signal.SIGALRM, _cp_timeout_handler)
            signal.alarm(self.timeout)
        return cvxopt.solvers.cp(self)
#        dims = dict(l=0, q=[], s=[2])
#        h = cvxopt.matrix(np.zeros(4))
#        G = cvxopt.spmatrix(np.ones(4), [0,1,2,3], [0,2,2,1], size=(4, len(self.params0)))
#        with aux.CvxoptFrame(feastol=1e-5, reltol=1e-4, abstol=1e-4):
#            return cvxopt.solvers.cp(self, G, h, dims)


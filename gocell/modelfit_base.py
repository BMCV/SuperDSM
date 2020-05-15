import gocell.surface as surface
import numpy as np
import cvxopt
import sys

from math         import sqrt
from scipy.linalg import orth
from scipy        import ndimage
from scipy.sparse import csr_matrix, bmat as sparse_block, diags as sparse_diag, issparse


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
        svec = diagquad(self.A, xvec) + 2 * np.inner(xvec.T, self.b) + self.c + smooth_mat @ self.ξ
        return svec.reshape(x.shape[-xdim:]) if isinstance(x, np.ndarray) else svec
    
    @staticmethod
    def create_ellipsoid(ξ, center, halfaxis1_len, halfaxis2_len, U=None):
        ev = lambda half_length: (1. / np.square(half_length))
        if U is None: U = orth(np.random.randn(2, 2)) # random rotation matrix
        A  = U.dot(np.diag((ev(halfaxis1_len), ev(halfaxis2_len)))).dot(U.T)
        b  = A.dot(center)
        c  = np.inner(center, b) - 1
        return PolynomialModel(ξ, A, -b, c)

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
    
    def map_to_image_pixels(self, g, roi):
        g_max_coord, roi_max_coord = np.array(g.model.shape) - 1., np.array(roi.model.shape) - 1.
        G = np.diag(1. / roi_max_coord)
        v = -G.dot(roi.offset)
        A = G.dot(self.A).dot(G)
        b = G.dot(self.A.dot(v) + self.b)
        c = np.inner(v, self.A.dot(v)) + 2 * np.inner(self.b, v) + self.c
        return PolynomialModel(self.ξ, A, b, c)


def diagquad(A, X):
    """Computes the diagonal entries of `X' A X` quickly.
    
    See: http://stackoverflow.com/a/14759341/1444073
    """
    return np.einsum('ij,ij->i', np.dot(X.T, A), X.T)


def get_fg_measures(y_map, roi):
    y_map  = y_map * roi.mask
    fg_map = (y_map > 0)
    fg_center = ndimage.measurements.center_of_mass(fg_map)
    dx_map = roi.get_map(normalized=False) - np.array(fg_center)[:, None, None]
    fg_std = sqrt(((dx_map * fg_map) ** 2).sum() / fg_map.sum())
    return fg_center, fg_std


def render_gaussian(shape, center, std):
    x_map = surface.get_pixel_map(shape, normalized=False)
    return np.exp(-np.square(x_map - np.array(center)[:, None, None]).sum(axis=0) / (2 * np.square(std)))


def get_roi_weights(y_map, roi, std_factor=1):
    if np.isinf(std_factor):
        return np.ones(roi.mask.shape)
    else:
        fg_center, fg_std = get_fg_measures(y_map, roi)
        return render_gaussian(roi.model.shape, fg_center, fg_std * std_factor)


def _create_gaussian_kernel(sigma, shape=None, shape_multiplier=1):
    if abs(shape_multiplier - 1) > 0 and shape is not None: raise ValueError()
    if shape is None: shape = [round(1 + sigma * 4 * shape_multiplier)] * 2
    inp = np.zeros(shape)
    inp[shape[0] // 2, shape[1] // 2] = 1
    return ndimage.gaussian_filter(inp, sigma)


def _convmat(filter_mask, img_shape, row_mask=None, col_mask=None):
    assert filter_mask.ndim == 2 and filter_mask.shape[0] == filter_mask.shape[1]
    assert filter_mask.shape[0] % 2 == 1
    if row_mask is None: row_mask = np.ones(img_shape, bool)
    if col_mask is None: col_mask = np.ones(img_shape, bool)
    w = filter_mask.shape[0] // 2
    mat = np.empty((row_mask.sum(), col_mask.sum()))
    z = np.zeros(np.add(img_shape, 2 * w))
    mat_next_row_idx = 0
    for p in np.ndindex(img_shape):
        if not row_mask[p]: continue
        sect = np.s_[p[0] : p[0] + filter_mask.shape[0], p[1] : p[1] + filter_mask.shape[1]]
        z[sect] = filter_mask
        mat[mat_next_row_idx]  = z[w : w + img_shape[0], w : w + img_shape[1]][col_mask].reshape(-1)
        mat_next_row_idx += 1
        z[sect] = 0
    return mat


def _create_subsample_grid(mask, subsample):
    subsample_grid = np.zeros_like(mask)
    subsample_grid[::subsample, ::subsample] = True
    subsample_grid = np.logical_and(mask, subsample_grid)
    while True:
        distances = mask * ndimage.distance_transform_bf(~subsample_grid, metric='chessboard')
        outside = (distances >= subsample)
        if not outside.any(): break
        min_outside_distance = distances[outside].min()
        min_outside_pixel = np.asarray(np.where(distances == min_outside_distance)).T[0]
        subsample_grid[tuple(min_outside_pixel)] = True
    return subsample_grid


def _create_masked_smooth_matrix(kernel, mask, subsample=1):
    mask = mask[np.where(mask.any(axis=1))[0], :]
    mask = mask[:, np.where(mask.any(axis=0))[0]]
    subsample_grid = _create_subsample_grid(mask, subsample)
    M = _convmat(kernel, mask.shape, row_mask=mask, col_mask=np.logical_and(mask, subsample_grid))
    M_sums = M.sum(axis=1)
    M /= M_sums[:, None]
    assert (M.sum(axis=0) > 0).all() and (M.sum(axis=1) > 0).all()
    return M


class Energy:

    def __init__(self, y_map, roi, w_map, epsilon, rho, smooth_amount, smooth_subsample, gaussian_shape_multiplier, model_type=PolynomialModel.TYPE):
        self.roi = roi
        self.p   = None

        if smooth_amount < np.inf:
            psf = _create_gaussian_kernel(smooth_amount, shape_multiplier=gaussian_shape_multiplier)
            smooth_mat = _create_masked_smooth_matrix(psf, roi.mask, smooth_subsample)
        else:
            smooth_mat = np.empty((roi.mask.sum(), 0))
        self.smooth_mat = csr_matrix(smooth_mat)

        self.x = self.roi.get_map()[:, roi.mask]
        self.w = w_map[roi.mask]
        self.y = y_map[roi.mask]

        assert epsilon > 0, 'epsilon must be strictly positive'
        self.epsilon = epsilon

        assert rho >= 0, 'rho must be positive'
        self.rho = rho

        # pre-compute common terms occuring in the computation of the derivatives
        self.model_type = model_type
        self.q = model_type.compute_derivatives(self.x)
    
    def update_maps(self, params):
        if self.p is not None and all(self.p.array == params.array): return
        s = params.s(self.x, self.smooth_mat)
        self.p     = params
        self.t     = self.y * s
        self.theta = None # invalidate
        
        valid_t_mask = self.t >= -np.log(sys.float_info.max)
        self.h = np.empty_like(self.t)
        self.h[  valid_t_mask] = np.exp(-self.t[valid_t_mask])
        self.h[~ valid_t_mask] = np.NaN
    
    def update_theta(self):
        if self.theta is None:
            valid_h_mask = ~np.isnan(self.h)
            self.theta = -np.ones_like(self.t)
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
            objective2 = self.rho * np.sqrt(np.square(params.ξ) + self.epsilon).sum() / self.smooth_mat.shape[1]
        else:
            objective2 = 0
        return objective1 + objective2
    
    def grad(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        f = [None] * len(self.q)
        term1 = -self.theta * self.y
#        if (abs(term1)<1e-8).sum() / np.prod(term1.shape) > 0.3:
#            raise ValueError()
        for i in range(len(f)): f[i] = term1 * self.q[i]
        grad = np.array(list(map(lambda f: np.inner(self.w.flat, f.flat), f)))
        if self.smooth_mat.shape[1] > 0:
            grad2  = (self.w.reshape(-1)[None, :] @ self.smooth_mat.multiply(term1[:, None])).reshape(-1)
            grad2 += self.rho * (params.ξ / np.sqrt(np.square(params.ξ) + self.epsilon)) / self.smooth_mat.shape[1]
            grad   = np.concatenate([grad, grad2])
        return grad
    
    def hessian(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        gamma = self.theta - np.square(self.theta)
        n = len(self.q) + self.smooth_mat.shape[1]
        D1 = np.asarray([-self.y * qi for qi in self.q])
        D2 = self.smooth_mat.multiply(-self.y[:, None]).T
        D1_, D2_ = D1 * (gamma * self.w.reshape(-1))[None, :], D2.multiply((gamma * self.w.reshape(-1))[None, :])
        if self.smooth_mat.shape[1] > 0:
            H = sparse_block([
                [D1_ @ D1.T, D1_ @ D2.T],
                [D2_ @ D1.T, D2_ @ D2.T]])
            g = self.rho * (1 / np.sqrt(np.square(params.ξ) + self.epsilon) - np.square(params.ξ) / np.power(np.square(params.ξ) + self.epsilon, 1.5)) / self.smooth_mat.shape[1]
            assert np.allclose(0, g[g < 0])
            g[g < 0] = 0
            H += sparse_diag(np.concatenate([np.zeros(6), g]))
        else:
            H = D1_ @ D1.T
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


class CP:

    def __init__(self, energy, params0, cachesize=0, cachetest=None):
        self.params0 = params0
        self.gradient = Cache(cachesize, lambda p: (energy(p), cvxopt.matrix(energy.grad(p)).T), equality=cachetest)
        self.hessian  = Cache(cachesize, lambda p:  energy.hessian(p), equality=cachetest)
    
    def __call__(self, params=None, w=None):
        if params is None:
            return 0, cvxopt.matrix(self.params0)
        else:
            p  = np.array(params).reshape(-1)
            l, Dl = self.gradient(p)
            assert not np.isnan(p).any()
            assert not np.isnan(np.array(Dl)).any()
            if w is None:
                return l, Dl
            else:
                H = self.hessian(p)
                if issparse(H):
                    H = H.tocoo()
                    H = cvxopt.spmatrix(w[0] * H.data, H.row, H.col, size=H.shape)
                else:
                    assert not np.isnan(H).any()
                    H = cvxopt.matrix(w[0] * H)
                return l, Dl, H
    
    def solve(self):
        return cvxopt.solvers.cp(self)


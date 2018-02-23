import surface
import numpy as np
import cvxopt
import sys

from math         import sqrt
from scipy.linalg import orth
from scipy        import ndimage


class PolynomialModelType:
    
    def get_model(self, params):
        return params if isinstance(params, PolynomialModel) else PolynomialModel(params)
    
    def compute_derivatives(self, x_map):
        derivatives = [None] * 6
        derivatives[0] = np.square(x_map[0])
        derivatives[1] = np.square(x_map[1])
        derivatives[2] = 2 * np.prod([x_map[i] for i in xrange(2)], axis=0)
        derivatives[3] = 2 * x_map[0]
        derivatives[4] = 2 * x_map[1]
        derivatives[5] = 1
        return derivatives


class PolynomialModel:
    
    TYPE = PolynomialModelType()
    
    def __init__(self, *args):
        if len(args) == 1 and len(args[0]) == 6:
            self.array = args[0].astype(float).flatten()
            self.a = args[0].flat[:3  ]
            self.b = args[0].flat[ 3:5]
            self.c = args[0].flat[   5]
        else:
            self.a = np.array([1, 1, 0]) if len(args) < 1 else args[0].flat[np.array([0, 3, 1])]
            self.b =         np.zeros(2) if len(args) < 2 else args[1].astype(float)
            self.c =                   0 if len(args) < 3 else float(args[2])
            self.array = np.concatenate([self.a, self.b, np.array([self.c])])

    def copy(self):
        return PolynomialModel(self.array.copy())
    
    @property
    def A(self):
        return np.array([self.a[0], self.a[2], self.a[2], self.a[1]]).reshape((2, 2))
    
    def s(self, x):
        xdim = x.ndim - 1 if isinstance(x, np.ndarray) else 0
        xvec = np.array(x).reshape((2, -1))
        svec = diagquad(self.A, xvec) + 2 * np.inner(xvec.T, self.b) + self.c
        return svec.reshape(x.shape[-xdim:]) if isinstance(x, np.ndarray) else svec
    
    @staticmethod
    def create_ellipsoid(center, halfaxis1_len, halfaxis2_len, U=None):
        ev = lambda half_length: (1. / np.square(half_length))
        if U is None: U = orth(np.random.randn(2, 2)) # random rotation matrix
        A  = U.dot(np.diag((ev(halfaxis1_len), ev(halfaxis2_len)))).dot(U.T)
        b  = A.dot(center)
        c  = np.inner(center, b) - 1
        return PolynomialModel(A, -b, c)
    
    def measure(self):
        """Returns the center `b` and halfaxes `H` of an ellipse-shaped model.
        
        The matrix `H` consists of two rows, which correspond to the two
        halfaxes. Thus, the two halfaxes can be accessed as `H[0]` and `H[1]`.
        """
        b = -np.linalg.inv(self.A + 1e-12 * np.eye(self.A.ndim)).dot(self.b)
        c = self.c - np.inner(b, self.A.dot(b))
        d,U = np.linalg.eigh(self.A)
        l = sqrt(np.max((np.zeros_like(d), -c / d), axis=0))
        L = U.dot(np.diag(l))
        return b, L.T
    
    def map_to_image_pixels(self, g, roi):
        g_max_coord, roi_max_coord = np.array(g.model.shape) - 1., np.array(roi.model.shape) - 1.
        G = np.diag(1. / roi_max_coord)
        v = -G.dot(roi.offset)
        A = G.dot(self.A).dot(G)
        b = G.dot(self.A.dot(v) + self.b)
        c = np.inner(v, self.A.dot(v)) + 2 * np.inner(self.b, v) + self.c
        return PolynomialModel(A, b, c)


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
        return ones(roi.mask.shape)
    else:
        fg_center, fg_std = get_fg_measures(y_map, roi)
        return render_gaussian(roi.model.shape, fg_center, fg_std * std_factor)


class Energy:

    def __init__(self, y_map, roi, w_map, r_map=None, kappa=5, model_type=PolynomialModel.TYPE):
        self.kappa = kappa if r_map is not None else 0
        self.roi   = roi
        self.p     = None
        self.x_map = self.roi.get_map()[:, roi.mask]
        self.w_map = w_map[roi.mask]
        self.r_map = (r_map if r_map is not None else np.zeros(w_map.shape, 'uint8'))[roi.mask]
        self.y_map = y_map[roi.mask]

        # pre-compute common terms occuring in the computation of the derivatives
        self.model_type = model_type
        self.q = model_type.compute_derivatives(self.x_map)
    
    def update_maps(self, params):
        if self.p is not None and all(self.p.array == params.array): return
        self.p     = params
        self.t_map = self.y_map * params.s(self.x_map)
        self.theta = None # invalidate
        
        valid_t_mask = self.t_map >= -np.log(sys.float_info.max)
        self.h_map   = np.empty_like(self.t_map)
        self.h_map[  valid_t_mask] = np.exp(-self.t_map[valid_t_mask])
        self.h_map[~ valid_t_mask] = np.NaN
        
        self.rs_map = self.r_map * params.s(self.x_map)
    
    def update_theta(self):
        if self.theta is None:
            valid_h_mask = ~np.isnan(self.h_map)
            self.theta = -np.ones_like(self.t_map)
            self.theta[valid_h_mask] = self.h_map[valid_h_mask] / (1 + self.h_map[valid_h_mask])
    
    def __call__(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        valid_h_mask = ~np.isnan(self.h_map)
        phi = np.zeros_like(self.t_map)
        phi[ valid_h_mask] = np.log(1 + self.h_map[valid_h_mask])
        phi[~valid_h_mask] = -self.t_map[~valid_h_mask]
        return np.inner(self.w_map.flat, phi.flat) + self.kappa * np.inner(self.w_map.flat, np.square(self.rs_map).flat)
    
    def grad(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        f_maps = [None] * len(self.q)
        for i in xrange(len(f_maps)): f_maps[i] = -self.theta * self.y_map * self.q[i]
        grad = np.array(map(lambda f_map: np.inner(self.w_map.flat, f_map.flat), f_maps))
        for i in xrange(len(grad)):
            grad[i] += self.kappa * np.inner(self.w_map.flat, (2 * self.rs_map * self.r_map * self.q[i]).flat)
        return grad
    
    def hessian(self, params):
        params = self.model_type.get_model(params)
        self.update_maps(params)
        self.update_theta()
        gamma = np.square(self.y_map) * (self.theta - np.square(self.theta)) + 2 * self.kappa * np.square(self.r_map)
        H = np.empty((len(self.q), len(self.q)))
        H_ik = lambda i, k: np.inner(self.w_map.flat, (gamma * self.q[i] * self.q[k]).flat)
        for i in xrange(H.shape[0]):
            H[i, i] = H_ik(i, i)
            for k in xrange(i + 1, H.shape[1]):
                value = H_ik(i, k)
                H[i, k] = value
                H[k, i] = value
        return H


class CP:

    def __init__(self, energy, params0, verbose=False, epsilon=0, scale=1):
        self.energy  = energy
        self.params0 = params0
        self.verbose = verbose
        self.epsilon = epsilon
        self.scale   = scale
    
    def __call__(self, params=None, w=None):
        if params is None:
            return 0, cvxopt.matrix(self.params0)
        else:
            p  = np.array(params).flatten()
            l  = self.scale * self.energy(p)
            Dl = cvxopt.matrix(self.scale * self.energy.grad(p)).T
            if w is None:
                return l, Dl
            else:
                H = self.scale * self.energy.hessian(p) + self.epsilon * np.eye(len(self.energy.q))
                if self.verbose: print(np.linalg.eigvals(H))
                return l, Dl, cvxopt.matrix(w[0] * H)
    
    def solve(self):
        return cvxopt.solvers.cp(self)


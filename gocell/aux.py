import gocell.mapper as mapper
import cvxopt, cvxopt.solvers
import sys
import numpy as np
import scipy.sparse
import warnings
import pathlib
import contextlib

from skimage.filters.rank import median as median_filter
from IPython.display      import clear_output


def set2str(S, delim=','):
    """Formats the set `S` as a string with the given delimiter.
    """
    return '{%s}' % (delim.join(str(s) for s in S))


def int_hist(g):
    """Computes the histogram of array `g`.
    """
    h, i = [], []
    for k in range(g.min(), g.max() + 1):
        hi = (g == k).sum()
        if hi > 0:
            i.append(k)
            h.append(hi)
    return i, h


def copy_dict(d):
    """Returns a copy of dict `d`.
    """
    assert isinstance(d, dict), 'not a "dict" object'
    return dict(d.items())


class Output:
    def __init__(self, parent=None, maxlen=np.inf):
        self.lines     = []
        self.current   = None
        self.parent    = parent
        self.maxlen    = maxlen
        self.truncated = 0
    
    def derive(self, maxlen=np.inf):
        child = Output(parent=self, maxlen=maxlen)
        if self.current is not None: child.lines.append(self.current)
        return child
    
    @staticmethod
    def get(out):
        return Output() if out is None else out
    
    def clear(self, flush=False):
        clear_output(not flush)
        p_list = [self]
        while p_list[-1].parent is not None:
            p_list += [p_list[-1].parent]
        for p in p_list[::-1]:
            if p.truncated > 0: print('[...] (%d)' % self.truncated)
            for line in p.lines: print(line)
        self.current = None

    def truncate(self, offset=0):
        if len(self.lines) + offset > self.maxlen:
            self.lines = self.lines[len(self.lines) + offset - self.maxlen:]
            self.truncated += 1
    
    def intermediate(self, line, flush=True):
        self.truncate(offset=+1)
        self.clear()
        self.current = line
        print(line)
        if flush: sys.stdout.flush()
    
    def write(self, line, keep_current=False):
        if keep_current and self.current is not None: self.lines.append(self.current)
        self.lines.append(line)
        self.truncate()
        self.clear()


def medianf(img, selem):
    img_min = img.min()
    img -= img_min
    img_max = img.max()
    img /= img_max
    return median_filter((img * 256).round().astype('uint8'), selem) * img_max / 256. + img_min


class CvxoptFrame:

    def __enter__(self):
        self.options = copy_dict(cvxopt.solvers.options)
        return self

    def __setitem__(self, key, value):
        cvxopt.solvers.options[key] = value

    def __exit__(self, *args):
        cvxopt.solvers.options.clear()
        cvxopt.solvers.options.update(self.options)


def threshold_gauss(data, tolerance, mode):
    X = np.array(list(data)).flat
    f = None
    if mode in ('l', 'lower'): f = -1
    if mode in ('u', 'upper'): f = +1
    assert f is not None, 'unknown mode "%s"' % mode
    X_std = np.std(X) if len(X) > 1 else np.inf
    t_gauss = np.mean(X) + f * X_std * tolerance
    assert not np.isnan(t_gauss)
    return t_gauss


def uplift_smooth_matrix(smoothmat, mask):
    assert mask.sum() == smoothmat.shape[0], 'smooth matrix and region mask are incompatible'
    if not scipy.sparse.issparse(smoothmat): warnings.warn(f'{uplift_smooth_matrix.__name__} received a dense matrix which is inefficient')
    M = scipy.sparse.coo_matrix((np.prod(mask.shape), smoothmat.shape[0]))
    M.data = np.ones(mask.sum())
    M.row  = np.where(mask.reshape(-1))[0]
    M.col  = np.arange(len(M.data))
    smoothmat2 = M.tocsr() @ smoothmat
    return smoothmat2


BUGFIX_ENABLED  = 1
BUGFIX_DISABLED = 0
BUGFIX_DISABLED_CRITICAL = -1


def is_bugfix_enabled(bugfix):
    if bugfix < 0:
        raise AssertionError(
            "Critical bugfix is disabled, aborting. " + \
            "Either enable the bugfix (set to BUGFIX_ENABLED) or mark it as non-critical (set to BUGFIX_DISABLED).")
    elif bugfix == 0:
        return False  ## disable the bugfix and proceed
    else:
        return True   ## enable the bugfix


def collapse_smooth_matrices(data):
    if 'processed_candidates' in data:
        for c in data['processed_candidates']:
            c.collapse_smooth_matrix()


def _restore_smooth_matrix(cidx, c, g, g_superpixels):
    with contextlib.redirect_stdout(None):
        c.restore_smooth_matrix(g, g_superpixels)
    return cidx, c


def restore_smooth_matrices(processes, data, out=None):
    if 'processed_candidates' in data:
        candidates = data['processed_candidates']
        out = Output.get(out)
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(processes, _restore_smooth_matrix,
                                                                 mapper.unroll(range(len(candidates))),
                                                                 mapper.unroll(candidates),
                                                                 data['g'], data['g_superpixels'])):
            out.intermediate(f'Restoring smooth matrix {ret_idx + 1} / {len(candidates)}... {100 * ret_idx / len(candidates):.1f} %')
            candidates[ret[0]].smooth_mat = ret[1].smooth_mat


def mkdir(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def join_path(path1, path2):
    return str(pathlib.Path(path1) / pathlib.Path(path2))


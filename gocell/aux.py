import cvxopt, cvxopt.solvers
import sys
import numpy as np
import scipy.sparse
import warnings
import pathlib
import contextlib
import ray
import fcntl, hashlib
import posix_ipc

from skimage.filters.rank import median as median_filter
from IPython.display      import clear_output


#def set2str(S, delim=','):
#    """Formats the set `S` as a string with the given delimiter.
#    """
#    return '{%s}' % (delim.join(str(s) for s in S))
#
#
#def int_hist(g):
#    """Computes the histogram of array `g`.
#    """
#    h, i = [], []
#    for k in range(g.min(), g.max() + 1):
#        hi = (g == k).sum()
#        if hi > 0:
#            i.append(k)
#            h.append(hi)
#    return i, h
#
#
def copy_dict(d):
    """Returns a copy of dict `d`.
    """
    assert isinstance(d, dict), 'not a "dict" object'
    return {item[0]: copy_dict(item[1]) if isinstance(item[1], dict) else item[1] for item in d.items()}


def is_jupyter_notebook():
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True
    except NameError: pass
    return False


def get_output(out=None):
    if out is not None:
        return out
    if is_jupyter_notebook():
        return JupyterOutput()
    else:
        return ConsoleOutput()


class JupyterOutput:

    def __init__(self, parent=None, maxlen=np.inf, muted=False):
        self.lines     = []
        self.current   = None
        self.parent    = parent
        self.maxlen    = maxlen
        self.truncated = 0
        self._muted    = muted

    @property
    def muted(self):
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, maxlen=np.inf):
        child = JupyterOutput(parent=self, maxlen=maxlen, muted=muted)
        if self.current is not None: child.lines.append(self.current)
        return child
    
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
        if self.muted: return
        self.truncate(offset=+1)
        self.clear()
        self.current = line
        print(line)
        if flush: sys.stdout.flush()
    
    def write(self, line, keep_current=False):
        if self.muted: return
        if keep_current and self.current is not None: self.lines.append(self.current)
        self.lines.append(line)
        self.truncate()
        self.clear()


class ConsoleOutput:
    def __init__(self, muted=False, parent=None, margin=0):
        self.parent = parent
        self._muted = muted
        self.margin = margin
    
    @staticmethod
    def get(out):
        return ConsoleOutput() if out is None else out

    def intermediate(self, line):
        if not self.muted:
            print(' ' * self.margin + line, end='\r')
            sys.stdout.flush()
    
    def write(self, line):
        if not self.muted:
            lines = line.split('\n')
            if len(lines) == 1:
                sys.stdout.write("\033[K");
                print(' ' * self.margin + line)
            else:
                for line in lines: self.write(line)

    @property
    def muted(self):
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, margin=0):
        assert margin >= 0
        return ConsoleOutput(muted, self, self.margin + margin)


#def medianf(img, selem):
#    img_min = img.min()
#    img -= img_min
#    img_max = img.max()
#    img /= img_max
#    return median_filter((img * 256).round().astype('uint8'), selem) * img_max / 256. + img_min


class CvxoptFrame:

    def __init__(self, **defaults):
        self._defaults  = copy_dict(defaults)
        self._interface = None

    def __enter__(self):
        self._interface = cvxopt.solvers.options
        self._options = copy_dict(self._interface)
        for key in self._defaults:
            self[key] = self._defaults[key]
        return self

    def __setitem__(self, key, value):
        self._interface[key] = value

    def __exit__(self, *args):
        self._interface.clear()
        self._interface.update(self._options)
        self._interface = None


#def threshold_gauss(data, tolerance, mode):
#    X = np.array(list(data)).flat
#    f = None
#    if mode in ('l', 'lower'): f = -1
#    if mode in ('u', 'upper'): f = +1
#    assert f is not None, 'unknown mode "%s"' % mode
#    X_std = np.std(X) if len(X) > 1 else np.inf
#    t_gauss = np.mean(X) + f * X_std * tolerance
#    assert not np.isnan(t_gauss)
#    return t_gauss


def uplift_smooth_matrix(smoothmat, mask):
    assert mask.sum() == smoothmat.shape[0], 'smooth matrix and region mask are incompatible'
    if not scipy.sparse.issparse(smoothmat): warnings.warn(f'{uplift_smooth_matrix.__name__} received a dense matrix which is inefficient')
    M = scipy.sparse.coo_matrix((np.prod(mask.shape), smoothmat.shape[0]))
    M.data = np.ones(mask.sum())
    M.row  = np.where(mask.reshape(-1))[0]
    M.col  = np.arange(len(M.data))
    smoothmat2 = M.tocsr() @ smoothmat
    return smoothmat2


def mkdir(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def join_path(path1, path2):
    return str(pathlib.Path(path1) / pathlib.Path(path2))


def is_subpath(path, subpath):
    if isinstance(   path, str):    path = pathlib.Path(   path)
    if isinstance(subpath, str): subpath = pathlib.Path(subpath)
    try:
        subpath.relative_to(path)
        return True
    except ValueError:
        return False


def get_ray_1by1(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids, num_returns=1)
        assert len(done) == 1
        yield ray.get(done[0])


def render_candidate_foregrounds(shape, candidates):
    foreground = np.zeros(shape, bool)
    for candidate in candidates:
        sel = candidate.fill_foreground(foreground)
        yield foreground
        foreground[sel].fill(False)


class SystemSemaphore:
    def __init__(self, name, limit):
        self.name  = name
        self.limit = limit

    @staticmethod
    def get_lock(lock):
        class NullLock:
            def __enter__(self): pass
            def __exit__ (self, _type, value, tb): pass
        if lock is None: return NullLock()
        else: return lock

    def __enter__(self):
        if np.isinf(self.limit):
            create_lock = lambda flags: None
        else:
            create_lock = lambda flags: posix_ipc.Semaphore(f'/{self.name}', flags, mode=384, initial_value=self.limit)
        self._lock = create_lock(posix_ipc.O_CREAT | posix_ipc.O_EXCL)
        class Lock:
            def __init__(self, create_lock):
                self._create_lock = create_lock

            def __enter__(self):
                self._lock = self._create_lock(posix_ipc.O_CREAT)
                if self._lock is not None: self._lock.acquire()

            def __exit__(self, _type, value, tb):
                if self._lock is not None: self._lock.release()
        return Lock(create_lock)

    def __exit__(self, _type, value, tb):
        if self._lock is not None: self._lock.unlink()


class SystemMutex:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        lock_id = hashlib.md5(self.name.encode('utf8')).hexdigest()
        self.fp = open(f'/tmp/.lock-{lock_id}.lck', 'wb')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def find_candidate_by_footprint(candidates, atom_labels, cmp='=='):
    result = []
    atom_labels = frozenset(atom_labels)
    for c in candidates:
        match = False

        if isinstance(cmp, str) and len(atom_labels & c.footprint) > 0 and eval(f'atom_labels {cmp} c.footprint'):
            match = True

        elif callable(cmp) and cmp(atom_labels, c.footprint):
            match = True

        if match:
            result.append(c)
            if cmp == '==': break
    return result


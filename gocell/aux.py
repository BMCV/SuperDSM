import cvxopt, cvxopt.solvers
import sys

from skimage.filter.rank import median as median_filter
from IPython.display     import clear_output


def set2str(S, delim=','):
    """Formats the set `S` as a string with the given delimiter.
    """
    return '{%s}' % (delim.join(str(s) for s in S))


def int_hist(g):
    """Computes the histogram of array `g`.
    """
    h, i = [], []
    for k in xrange(g.min(), g.max() + 1):
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
    def __init__(self, parent=None):
        self.lines = []
        self.current = None
        self.parent = parent
    
    def derive(self):
        child = Output(parent=self)
        if self.current is not None: child.lines.append(self.current)
        return child
    
    @staticmethod
    def get(out):
        return Output() if out is None else out
    
    def clear(self):
        clear_output(True)
        p = self.parent
        while p is not None:
            for line in p.lines: print(line)
            p = p.parent
        for line in self.lines: print(line)
        self.current = None
    
    def intermediate(self, line, flush=True):
        self.clear()
        self.current = line
        print(line)
        if flush: sys.stdout.flush()
    
    def write(self, line, keep_current=False):
        if keep_current and self.current is not None: self.lines.append(self.current)
        self.lines.append(line)
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


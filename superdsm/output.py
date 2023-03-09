import numpy as np
import sys

from IPython.display import clear_output


def is_jupyter_notebook():
    """Checks whether code is being executed in a Jupyter notebook.

    :return: ``True`` if code is being executed in a Jupyter notebook and ``False`` otherwise.
    """
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True
    except NameError: pass
    return False


def get_output(out=None):
    """Returns a suitable output.

    :param out: This will be returned if it is not ``None``.
    :return: A :py:class:`~.JupyterOutput` object if code is being executed in a Jupyter notebook and a :py:class:`~.ConsoleOutput` object otherwise.
    """
    if out is not None:
        return out
    if is_jupyter_notebook():
        return JupyterOutput()
    else:
        return ConsoleOutput()


class Text:

    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    DARKCYAN  = '\033[36m'
    BLUE      = '\033[94m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'

    @staticmethod
    def style(text, style):
        return f'{style}{text}{Text.END}'


class JupyterOutput:

    def __init__(self, parent=None, maxlen=np.inf, muted=False, margin=0):
        assert margin >= 0
        self.lines     = []
        self.current   = None
        self.parent    = parent
        self.maxlen    = maxlen
        self.truncated = 0
        self._muted    = muted
        self.margin    = margin

    @property
    def muted(self):
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, maxlen=np.inf, margin=0):
        child = JupyterOutput(parent=self, maxlen=maxlen, muted=muted, margin=margin)
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
        line = ' ' * self.margin + line
        self.truncate(offset=+1)
        self.clear()
        self.current = line
        print(line)
        if flush: sys.stdout.flush()
    
    def write(self, line, keep_current=False):
        if self.muted: return
        if keep_current and self.current is not None: self.lines.append(self.current)
        line = ' ' * self.margin + line
        self.lines.append(line)
        self.truncate()
        self.clear()


class ConsoleOutput:
    def __init__(self, muted=False, parent=None, margin=0):
        self.parent = parent
        self._muted = muted
        self.margin = margin
        self._intermediate_line_length = 0
    
    @staticmethod
    def get(out):
        return ConsoleOutput() if out is None else out

    def intermediate(self, line):
        if not self.muted:
            _line = ' ' * self.margin + line
            print(self._finish_line(_line), end='\r')
            self._intermediate_line_length = len(_line)
            sys.stdout.flush()

    def _finish_line(self, line):
        return line + ' ' * max((0, self._intermediate_line_length - len(line)))
    
    def write(self, line):
        if not self.muted:
            lines = line.split('\n')
            if len(lines) == 1:
                sys.stdout.write("\033[K")
                print(' ' * self.margin + line)
            else:
                for line in lines: self.write(line)

    @property
    def muted(self):
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, margin=0):
        assert margin >= 0
        return ConsoleOutput(muted, self, self.margin + margin)
        
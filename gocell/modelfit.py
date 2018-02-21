import aux
import mapper
import cvxopt


class Frame:

    def __enter__(self):
        self.options = aux.copy_dict(cvxopt.solvers.options)
        return self

    def __setitem__(self, key, value):
        cvxopt.solvers.options[key] = value

    def __exit__(self, *args):
        for key in cvxopt.solvers.options: del cvxopt.solvers.options[k]
        for key in self.options: cvxopt.solvers.options[key] = self.options[key]


def process_candidate():
    assert False, 'not implemented yet'


def fork_based_backend(num_forks):
    def _imap(g, unique_candidates, g_superpixels, modelfit_kwargs, out):
        for ret_idx, ret in enumerate(mapper.fork.imap_unordered(num_forks,
                                                                 process_candidate,
                                                                 mapper.unroll(xrange(len(unique_candidates))),
                                                                 g, g_superpixels,
                                                                 mapper.unroll(unique_candidates),
                                                                 modelfit_kwargs)):
            out.intermediate('Processed candidate %d / %d (using %d forks)' % \
                (ret_idx + 1, len(unique_candidates), num_forks))
            yield ret
    return _imap


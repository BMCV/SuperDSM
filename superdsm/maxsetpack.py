from .output import get_output

import numpy as np
import cvxpy as cp
import scipy.sparse


def solve_maxsetpack(objects, out=None):
    accepted_objects  = []  ## primal variable
    remaining_objects = list(objects)

    out = get_output(out)
    w = lambda c: c.energy
    while len(remaining_objects) > 0:

        # choose the best remaining object
        best_object = max(remaining_objects, key=w)
        accepted_objects.append(best_object)

        # discard conflicting objects
        remaining_objects = [c for c in remaining_objects if len(c.footprint & best_object.footprint) == 0]

    out.write(f'MAXSETPACK - GREEDY accepted objects: {len(accepted_objects)}')
    return accepted_objects


def solve_maxsetpack_lp(objects, out=None):
    out = get_output(out)
    n = len(objects)
    m = 1 + max(max(c.footprint) for c in objects)
    u = cp.Variable(n)
    w = np.asarray([c.energy for c in objects]).reshape(n, 1)
    A = []
    for c_idx, c in enumerate(objects):
        A += [(atom_label, c_idx) for atom_label in c.footprint]
    A = scipy.sparse.coo_matrix((np.ones(len(A)), np.transpose(A)), shape=(m, n))
    prob = cp.Problem(cp.Maximize(w.T @ u), [A @ u <= np.ones(m)])
    prob.solve()
    return prob.value


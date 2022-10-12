import ._aux as aux
import numpy as np
import cvxpy as cp
import scipy.sparse


def solve_maxsetpack(candidates, out=None):
    accepted_candidates  = []  ## primal variable
    remaining_candidates = list(candidates)

    out = gocell.aux.get_output(out)
    w = lambda c: c.energy
    while len(remaining_candidates) > 0:

        # choose the best remaining candidate
        best_candidate = max(remaining_candidates, key=w)
        accepted_candidates.append(best_candidate)

        # discard conflicting candidates
        remaining_candidates = [c for c in remaining_candidates if len(c.footprint & best_candidate.footprint) == 0]

    out.write(f'MAXSETPACK - GREEDY accepted candidates: {len(accepted_candidates)}')
    return accepted_candidates


def solve_maxsetpack_lp(candidates, out=None):
    out   = gocell.aux.get_output(out)
    n = len(candidates)
    m = 1 + max(max(c.footprint) for c in candidates)
    u = cp.Variable(n)
    w = np.asarray([c.energy for c in candidates]).reshape(n, 1)
    A = []
    for c_idx, c in enumerate(candidates):
        A += [(atom_label, c_idx) for atom_label in c.footprint]
    A = scipy.sparse.coo_matrix((np.ones(len(A)), np.transpose(A)), shape=(m, n))
    prob = cp.Problem(cp.Maximize(w.T @ u), [A @ u <= np.ones(m)])
    prob.solve()
    return prob.value


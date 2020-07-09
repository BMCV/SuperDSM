import gocell.pipeline as pipeline
import gocell.aux      as aux
import gocell.config   as config
import cvxopt, cvxopt.solvers
import numpy as np
import math


class MinSetCoverWeights(pipeline.Stage):

    def __init__(self):
        super(MinSetCoverWeights, self).__init__('min_setcover_weights',
                                                 inputs  = ['processed_candidates'],
                                                 outputs = ['min_setcover_weights'])

    def process(self, input_data, cfg, out, log_root_dir):
        candidates = input_data['processed_candidates']
        alpha = float(config.get_value(cfg, 'alpha', 1))
        alpha_scale = config.get_value(cfg, 'alpha_scale', 'median')

        if len(candidates) == 0 or alpha_scale == 'constant':
            alpha_scale = 1
        else:
            energies = [c.energy for c in candidates]
            if alpha_scale == 'mean':
                alpha_scale = np.mean(energies)
            elif alpha_scale == 'median':
                alpha_scale = np.median(energies)
            else:
                raise Valueerror('unknown alpha_scale: "%s"' % alpha_scale)
        weights = dict((c, c.energy + alpha_scale * alpha) for c in candidates)

        out.write('Computed MINSETCOVER weights')

        return {
            'min_setcover_weights': weights
        }


class MinSetCoverGreedy(pipeline.Stage):

    def __init__(self):
        super(MinSetCoverGreedy, self).__init__('min_setcover_greedy',
                                                inputs  = ['processed_candidates', 'g_superpixels', 'min_setcover_weights'],
                                                outputs = ['accepted_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        candidates, g_superpixels, w = input_data['processed_candidates'], input_data['g_superpixels'], input_data['min_setcover_weights']
        accepted_candidates = []  ## primal variable

        remaining_candidates  = list(candidates)
        uncovered_superpixels = set(g_superpixels.flatten())
        while len(remaining_candidates) > 0:

            # compute prices of remaining candidates
            prices = dict((c, w[c] / len(c.superpixels & uncovered_superpixels)) for c in remaining_candidates)
            
            # choose the best remaining candidate
            best_candidate = min(prices, key=prices.get)
            accepted_candidates.append(best_candidate)

            # discard conflicting candidates
            uncovered_superpixels -= best_candidate.superpixels
            remaining_candidates   = [c for c in remaining_candidates if len(c.superpixels & uncovered_superpixels) > 0]

        out.write('Greedy MINSETCOVER - Accepted candidates: %d' % len(accepted_candidates))

        if config.get_value(cfg, 'merge_step', True):
            replacements_count = 0
            for c_new in sorted(candidates, key=lambda c: w[c]):
                if c_new in accepted_candidates: continue
                valid_replacement, blockers = True, set()
                for c in accepted_candidates:
                    overlap = len(c.superpixels & c_new.superpixels)
                    if overlap == 0: continue
                    if overlap < len(c.superpixels):
                        valid_replacement = False
                        break
                    assert overlap == len(c.superpixels)
                    blockers |= {c}
                if not valid_replacement: continue
                if w[c_new] < sum(w[c] for c in blockers):
                    replacements_count += len(blockers)
                    accepted_candidates = [c for c in accepted_candidates if c not in blockers] + [c_new]

            out.write('Greedy MINSETCOVER - Merged candidates: %d' % replacements_count)

        return {
            'accepted_candidates': accepted_candidates
        }


class MinSetCoverCheck(pipeline.Stage):

    def __init__(self):
        super(MinSetCoverCheck, self).__init__('min_setcover_check',
                                               inputs  = ['g_superpixels', 'processed_candidates', 'accepted_candidates', 'min_setcover_weights'],
                                               outputs = ['min_setcover_min_accuracy'])

    def process(self, input_data, cfg, out, log_root_dir):
        accepted_candidates  = input_data[ 'accepted_candidates']
        min_setcover_weights = input_data['min_setcover_weights']

        apx_primal = sum(min_setcover_weights[c] for c in accepted_candidates)
        try:
            opt_dual = MinSetCoverCheck.solve_dual_lp_relaxation(input_data)

            assert apx_primal >= opt_dual or abs(apx_primal - opt_dual) < 1e-4 * opt_dual
            apx_primal = max((apx_primal, opt_dual))

            min_accuracy = opt_dual / apx_primal if apx_primal > 0 else 0.
            out.write('Minimum accuracy of MINSETCOVER solution: %5.2f %%' % (100 * min_accuracy))

        except Exception as err:
            out.write('Minimum accuracy of MINSETCOVER -- Failure: %s' % repr(err))
            min_accuracy = 0

        return {
            'min_setcover_min_accuracy': min_accuracy
        }

    @staticmethod
    def solve_dual_lp_relaxation(input_data):
        superpixels = list(set(input_data['g_superpixels'].flatten()) - {0})
        min_setcover_weights = input_data['min_setcover_weights']
        max_weight = float(max(min_setcover_weights.values()))
        if max_weight == 0: max_weight = 1

        # non-negativity constraints:
        G = [ -np.eye(len(superpixels))]
        h = [np.zeros(len(superpixels))]

        # packing constraints:
        for c in input_data['processed_candidates']:
            G_row = np.zeros((1, len(superpixels)))
            for s in c.superpixels:
                i = superpixels.index(s)
                G_row[0, i] = 1
            G.append(G_row)
            h.append(np.array([min_setcover_weights[c] / max_weight]))

        assert all(G_row.sum() >= 1 for G_row in np.array(G)[len(superpixels):]), 'failed to build LP'
        G = cvxopt.matrix(np.concatenate(G, axis=0))
        h = cvxopt.matrix(np.concatenate(h, axis=0))
        with aux.CvxoptFrame() as batch:
            batch['show_progress'] = False
            batch['abstol'] = min((1e-7, 1 / max_weight))
            batch['reltol'] = min((1e-6, 1 / max_weight))
            solution = cvxopt.solvers.lp(cvxopt.matrix(-np.ones(len(superpixels))), G, h)
        assert solution['status'] == 'optimal', 'failed to find optimal LP solution'
        return solution['primal objective'] * (-1) * max_weight


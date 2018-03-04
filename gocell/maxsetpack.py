import pipeline
import aux
import config
import cvxopt, cvxopt.solvers
import numpy as np
import math


class MaxSetPackWeights(pipeline.Stage):

    def __init__(self):
        super(MaxSetPackWeights, self).__init__('max_setpack_weights',
                                                inputs  = ['processed_candidates', 'superpixels_covered_by'],
                                                outputs = ['max_setpack_weights'])

    def process(self, input_data, cfg, out):
        candidates, superpixels_covered_by = input_data['processed_candidates'], input_data['superpixels_covered_by']
        alpha = float(config.get_value(cfg, 'alpha',  1.  ))
        beta  = float(config.get_value(cfg, 'beta' ,  1e-8))
        gamma = float(config.get_value(cfg, 'gamma',  1.  ))
        form  =       config.get_value(cfg, 'form' , 'add')

        energies = [candidate.energy for candidate in candidates]
        high_energy = np.mean(energies) + np.std(energies)

        weights = []
        for cidx, candidate in enumerate(candidates):

            if form == 'add':
                weights.append(candidate.energy + high_energy * alpha / (beta + len(superpixels_covered_by[candidate])))
            elif form == 'mult':
                weights.append(candidate.energy / math.pow(beta + gamma * len(superpixels_covered_by[candidate]), alpha))
            else:
                raise ValueError('unknown "form" parameter: %s' % form)

            out.intermediate('Computed MAXSETPACK weight %d / %d' % (cidx + 1, len(candidates)))
        out.write('Computed MAXSETPACK weights')
        max_weight = max(weights)
        weights_dict = dict([(candidate, max_weight - weight) for candidate, weight in zip(candidates, weights)])

        return {
            'max_setpack_weights': weights_dict
        }


class MaxSetPackGreedy(pipeline.Stage):

    def __init__(self):
        super(MaxSetPackGreedy, self).__init__('max_setpack_greedy',
                                               inputs  = ['processed_candidates', 'max_setpack_weights'],
                                               outputs = ['accepted_candidates'])

    def process(self, input_data, cfg, out):
        candidates, weights = input_data['processed_candidates'], input_data['max_setpack_weights']
        accepted_candidates = []  ## primal variable
#        superpixel_charges  = {}  ## dual variables used to compute the accuracy of the approximation
#
#        while len(weights) > 0:
#            
#            # choose the best remaining candidate
#            best_candidate = max(weights, key=weights.get)
#            accepted_candidates.append(best_candidate)
#
#            # update dual variable
#            superpixel_charges[list(best_candidate.superpixels)[0]] = weights[best_candidate]
#
#            # discard conflicting candidates
#            conflicting = [c for c in weights.keys() if len(c.superpixels & best_candidate.superpixels) > 0]
#            for c in sorted(conflicting, key=weights.get, revert=True):
#                if not any(s in superpixel_charges for s in c.superpixels):
#                    superpixel_charges[list(c.superpixels)[0]] = weights[c]
#            weights = dict([(c, w) for c, w in weights.items() if c not in conflicting)

        while len(weights) > 0:
            
            # choose the best remaining candidate
            best_candidate = max(weights, key=weights.get)
            accepted_candidates.append(best_candidate)

            # discard conflicting candidates
            weights = dict([(c, w) for c, w in weights.items() if len(c.superpixels & best_candidate.superpixels) == 0])        

            out.intermediate('Greedy MAXSETPACK - Remaining candidates: %d' % len(weights))
        out.write('Greedy MAXSETPACK - Accepted candidates: %d' % len(accepted_candidates))

        return {
            'accepted_candidates': accepted_candidates
        }


class MaxSetPackCheck(pipeline.Stage):

    def __init__(self):
        super(MaxSetPackCheck, self).__init__('max_setpack_check',
                                              inputs  = ['g_superpixels', 'processed_candidates', 'accepted_candidates', 'max_setpack_weights'],
                                              outputs = ['max_setpack_min_accuracy'])

    def process(self, input_data, cfg, out):
        accepted_candidates = input_data['accepted_candidates']
        max_setpack_weights = input_data['max_setpack_weights']

        apx_primal = sum(max_setpack_weights[c] for c in accepted_candidates)
        opt_dual   = self.solve_dual_lp_relaxation(input_data)

        assert apx_primal <= opt_dual or abs(apx_primal - opt_dual) < 1e-4 * opt_dual
        apx_primal = min((apx_primal, opt_dual))

        min_accuracy = apx_primal / opt_dual if opt_dual > 0 else 0.
        out.write('Minimum accuracy of MAXSETPACK solution: %5.2f %%' % (100 * min_accuracy))

        return {
            'max_setpack_min_accuracy': min_accuracy
        }

    def solve_dual_lp_relaxation(self, input_data):
        superpixels = list(set(input_data['g_superpixels'].flatten()) - {0})
        max_setpack_weights = input_data['max_setpack_weights']
        max_weight = float(max(max_setpack_weights.values()))
        G = [ -np.eye(len(superpixels))]
        h = [np.zeros(len(superpixels))]
        for c in input_data['processed_candidates']:
            G_row = np.zeros((1, len(superpixels)))
            for s in c.superpixels:
                i = superpixels.index(s)
                G_row[0, i] = -1
            G.append(G_row)
            h.append(np.array([-max_setpack_weights[c] / max_weight]))
        assert all(G_row.sum() <= -1 for G_row in np.array(G))
        G = cvxopt.matrix(np.concatenate(G, axis=0))
        h = cvxopt.matrix(np.concatenate(h, axis=0))
        with aux.CvxoptFrame() as batch:
            batch['show_progress'] = False
            solution = cvxopt.solvers.lp(cvxopt.matrix(np.ones(len(superpixels))), G, h)
        assert solution['status'] == 'optimal'
        return solution['primal objective'] * max_weight


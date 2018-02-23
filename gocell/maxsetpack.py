import pipeline
import config


class MaxSetPackWeights(pipeline.Stage):

    def __init__(self):
        super(MaxSetPackWeights, self).__init__('max_setpack_weights',
                                                inputs  = ['processed_candidates', 'superpixels_covered_by'],
                                                outputs = ['max_setpack_weights'])

    def process(self, input_data, cfg, out):
        candidates, superpixels_covered_by = input_data['processed_candidates'], input_data['superpixels_covered_by']
        alpha = float(config.get_value(cfg, 'alpha', 1.  ))
        beta  = float(config.get_value(cfg, 'beta' , 1e-8))

        weights = []
        for cidx, candidate in enumerate(candidates):
            weights.append(candidate.energy + alpha / (beta + len(superpixels_covered_by[candidate])))
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
        accepted_candidates = []

        while len(weights) > 0:
            
            # choose the best remaining candidate
            best_candidate = max(weights, key=weights.get)
            accepted_candidates.append(best_candidate)

            # discard conflicting candidates
            for s in best_candidate.superpixels:
                weights = dict([(c, w) for c, w in weights.items() if len(c.superpixels & best_candidate.superpixels) == 0])

            out.intermediate('Greedy MAXSETPACK - Remaining candidates: %d' % len(weights))
        out.write('Greedy MAXSETPACK - Accepted candidates: %d' % len(accepted_candidates))

        return {
            'accepted_candidates': accepted_candidates
        }


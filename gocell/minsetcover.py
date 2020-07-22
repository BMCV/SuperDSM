import gocell.aux


def _solve_minsetcover(candidates, alpha, out=None):
    accepted_candidates  = []  ## primal variable
    remaining_candidates = list(candidates)
    uncovered_atoms      = set.union(*[c.footprint for c in candidates])

    out = gocell.aux.get_output(out)
    w = lambda c: c.energy + alpha
    while len(remaining_candidates) > 0:

        # compute prices of remaining candidates
        prices = dict((c, w(c) / len(c.footprint & uncovered_atoms)) for c in remaining_candidates)
        
        # choose the best remaining candidate
        best_candidate = min(prices, key=prices.get)
        accepted_candidates.append(best_candidate)

        # discard conflicting candidates
        uncovered_atoms -= best_candidate.footprint
        remaining_candidates = [c for c in remaining_candidates if len(c.footprint & uncovered_atoms) > 0]

    out.write(f'Greedy MINSETCOVER - Accepted candidates: {len(accepted_candidates)}')

    replacements_count = 0
    for c_new in sorted([c for c in candidates if c not in accepted_candidates], key=lambda c: w(c)):
        valid_replacement, blockers = True, set()
        for c in accepted_candidates:
            overlap = len(c.footprint & c_new.footprint)
            if overlap == 0: continue
            if overlap < len(c.footprint):
                valid_replacement = False
                break
            assert overlap == len(c.footprint)
            blockers |= {c}
        if not valid_replacement: continue
        if w(c_new) < sum(w(c) for c in blockers):
            replacements_count += len(blockers)
            accepted_candidates = [c for c in accepted_candidates if c not in blockers] + [c_new]

    out.write(f'Greedy MINSETCOVER - Merged candidates: {replacements_count}')

    return accepted_candidates


def _get_atom_label(atom):
    assert len(atom.footprint) == 1
    return list(atom.footprint)[0]


class MinSetCover:

    def __init__(self, atoms, alpha, adjacencies):
        self.atoms = {_get_atom_label(atom): atom for atom in atoms}
        self.alpha = alpha
        self.adjacencies = adjacencies
        self.candidates_by_cluster = {cluster: [atom for atom in atoms if adjacencies.get_cluster_label(_get_atom_label(atom)) == cluster] for cluster in adjacencies.cluster_labels}
        self.  solution_by_cluster = {cluster: self.candidates_by_cluster[cluster] for cluster in adjacencies.cluster_labels}

    def get_atom(self, atom_label):
        return self.atoms[atom_label]

    def _update_partial_solution(self, cluster_label, out):
        candidates = self.candidates_by_cluster[cluster_label]
        partial_solution = _solve_minsetcover(candidates, self.alpha, out)
        self.solution_by_cluster[cluster_label] = partial_solution

    def update(self, new_candidates, out=None):
        invalidated_clusters = []
        for new_candidate in new_candidates:
            cluster_label = self.adjacencies.get_cluster_label(list(new_candidate.footprint)[0])
            invalidated_clusters.append(cluster_label)
            self.candidates_by_cluster[cluster_label].append(new_candidate)
        for cluster_label in frozenset(invalidated_clusters):
            self._update_partial_solution(cluster_label, out)

    def get_cluster_costs(self, cluster_label):
        partial_solution = self.solution_by_cluster[cluster_label]
        return sum(c.energy for c in partial_solution) + self.alpha * len(partial_solution)

    @property
    def solution(self):
        return sum((list(partial_solution) for partial_solution in self.solution_by_cluster.values()), [])

    @property
    def costs(self):
        solution = self.solution
        return sum(c.energy for c in solution) + self.alpha * len(solution)


#class MinSetCoverCheck(pipeline.Stage):
#
#    def __init__(self):
#        super(MinSetCoverCheck, self).__init__('min_setcover_check',
#                                               inputs  = ['g_superpixels', 'processed_candidates', 'accepted_candidates', 'min_setcover_weights'],
#                                               outputs = ['min_setcover_min_accuracy'])
#
#    def process(self, input_data, cfg, out, log_root_dir):
#        accepted_candidates  = input_data[ 'accepted_candidates']
#        min_setcover_weights = input_data['min_setcover_weights']
#
#        apx_primal = sum(min_setcover_weights[c] for c in accepted_candidates)
#        try:
#            opt_dual = MinSetCoverCheck.solve_dual_lp_relaxation(input_data)
#
#            assert apx_primal >= opt_dual or abs(apx_primal - opt_dual) < 1e-4 * opt_dual
#            apx_primal = max((apx_primal, opt_dual))
#
#            min_accuracy = opt_dual / apx_primal if apx_primal > 0 else 0.
#            out.write('Minimum accuracy of MINSETCOVER solution: %5.2f %%' % (100 * min_accuracy))
#
#        except Exception as err:
#            out.write('Minimum accuracy of MINSETCOVER -- Failure: %s' % repr(err))
#            min_accuracy = 0
#
#        return {
#            'min_setcover_min_accuracy': min_accuracy
#        }
#
#    @staticmethod
#    def solve_dual_lp_relaxation(input_data):
#        superpixels = list(set(input_data['g_superpixels'].flatten()) - {0})
#        min_setcover_weights = input_data['min_setcover_weights']
#        max_weight = float(max(min_setcover_weights.values()))
#        if max_weight == 0: max_weight = 1
#
#        # non-negativity constraints:
#        G = [ -np.eye(len(superpixels))]
#        h = [np.zeros(len(superpixels))]
#
#        # packing constraints:
#        for c in input_data['processed_candidates']:
#            G_row = np.zeros((1, len(superpixels)))
#            for s in c.superpixels:
#                i = superpixels.index(s)
#                G_row[0, i] = 1
#            G.append(G_row)
#            h.append(np.array([min_setcover_weights[c] / max_weight]))
#
#        assert all(G_row.sum() >= 1 for G_row in np.array(G)[len(superpixels):]), 'failed to build LP'
#        G = cvxopt.matrix(np.concatenate(G, axis=0))
#        h = cvxopt.matrix(np.concatenate(h, axis=0))
#        with aux.CvxoptFrame() as batch:
#            batch['show_progress'] = False
#            batch['abstol'] = min((1e-7, 1 / max_weight))
#            batch['reltol'] = min((1e-6, 1 / max_weight))
#            solution = cvxopt.solvers.lp(cvxopt.matrix(-np.ones(len(superpixels))), G, h)
#        assert solution['status'] == 'optimal', 'failed to find optimal LP solution'
#        return solution['primal objective'] * (-1) * max_weight


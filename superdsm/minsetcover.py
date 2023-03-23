from .output import get_output


def _merge_minsetcover(objects, accepted_objects, beta):
    replacements_count = 0
    w = lambda c: c.energy + beta
    for c_new in sorted([c for c in objects if c not in accepted_objects], key=lambda c: w(c)):
        valid_replacement, blockers = True, set()
        for c in accepted_objects:
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
            accepted_objects = [c for c in accepted_objects if c not in blockers] + [c_new]
    return accepted_objects, replacements_count


def _solve_minsetcover(objects, beta, merge=True, out=None):
    accepted_objects  = []  ## primal variable
    remaining_objects = list(objects)
    uncovered_atoms      = set.union(*[c.footprint for c in objects])

    out = get_output(out)
    w = lambda c: c.energy + beta
    while len(remaining_objects) > 0:

        # compute prices of remaining objects
        prices = dict((c, w(c) / len(c.footprint & uncovered_atoms)) for c in remaining_objects)
        
        # choose the best remaining object
        best_object = min(prices, key=prices.get)
        accepted_objects.append(best_object)

        # discard conflicting objects
        uncovered_atoms -= best_object.footprint
        remaining_objects = [c for c in remaining_objects if len(c.footprint & uncovered_atoms) > 0]

    out.write(f'MINSETCOVER - GREEDY accepted objects: {len(accepted_objects)}')

    if merge:
        accepted_objects, replacements_count = _merge_minsetcover(objects, accepted_objects, beta)
        out.write(f'MINSETCOVER - MERGED objects: {replacements_count}')

    return accepted_objects


DEFAULT_TRY_LOWER_BETA = 4
DEFAULT_LOWER_BETA_MUL = 0.8


def solve_minsetcover(objects, beta, merge=True, try_lower_beta=DEFAULT_TRY_LOWER_BETA, lower_beta_mul=DEFAULT_LOWER_BETA_MUL, out=None):
    """Computs an approximative min-weight set-cover.

    This function implements Algorithm 2 of the :ref:`paper <references>`.

    :param objects: Corresponds to the family of the *candidate* sets :math:`\\mathscr S`. Any candidate set :math:`X \\in \\mathscr S` is either included in :math:`\\mathscr X` or not. Must be a list of objects, so that ``c.energy`` correspsonding to :math:`c(X)` and ``c`` is of the class :py:class:`~.objects.Object`.
    :param beta: The sparsity parameter :math:`\\beta \\geq 0`.
    :param merge: The *merge step* of Algorithm 2 will be used only if ``True`` is passed.
    :param try_lower_beta: The number of *repetitions* to perform after the initial iteration (this is the *max_iter* parameter of Algorithm 2 minus 1). Each additional repetitions uses a more conservative merging strategy (i.e. the sparsity parameter :math:`\\beta` is reduced).
    :param lower_beta_mul: The factor used to reduce the sparsity parameter :math:`\\beta` in each repetition (this is the parameter :math:`\\gamma` of Algorithm 2, where :math:`0 < \\gamma < 1`).
    :param out: An instance of an :py:class:`~superdsm.output.Output` sub-class, ``'muted'`` if no output should be produced, or ``None`` if the default output should be used.
    :return: The min-weight set-cover :math:`\\mathscr X \\subseteq \\mathscr S`.
    """
    assert beta >= 0
    assert 0 < lower_beta_mul < 1
    out = get_output(out)
    solution1 = _solve_minsetcover(objects, beta, merge, out)
    if try_lower_beta > 0 and beta > 0:
        new_beta = beta * lower_beta_mul
        out.write(f'MINSETCOVER retry with lower beta: {new_beta:g}')
        solution2 = solve_minsetcover(objects, new_beta, merge, try_lower_beta - 1, lower_beta_mul, out)
        solution1_value = sum(c.energy for c in solution1) + beta * len(solution1)
        solution2_value = sum(c.energy for c in solution2) + beta * len(solution2)
        if solution2_value < solution1_value:
            out.write(f'MINSETCOVER solution for beta={beta:g} improved by {solution2_value - solution1_value:,g} (-{100 * (1 - solution2_value / solution1_value):.2f}%)')
            return solution2
    return solution1


def _get_atom_label(atom):
    assert len(atom.footprint) == 1
    return list(atom.footprint)[0]


class MinSetCover:
    """Objects of this class represent solved instances of the min-weight set-cover problem.

    The objective of the problem is to determine a sparse minimal-energy family :math:`\\mathscr X` of sets, where :math:`\\nu(X)` is the energy of a set :math:`X`, and :math:`\\beta \\geq 0` is the sparsity parameter,

    .. math:: \\operatorname{MSC}(\\mathscr S) = \\min_{\\mathscr X \\subseteq \\mathscr S} \\sum_{X \\in \\mathscr X} \\beta + \\nu(X) \\enspace\\text{s.t. } \\bigcup \\mathscr S = \\bigcup \\mathscr X,

    where the sparse minimal-energy family :math:`\\mathscr X` is a *min-weight set-cover*. See :ref:`pipeline_theory_jointsegandclustersplit` and Section 2.3.2 in the :ref:`paper <references>` for details.

    The family of candidate sets :math:`\\mathscr S` is empty initially. Object prototypes (i.e. sets of atomic image regions) are added to :math:`\\mathscr S` by the :py:meth:`~.update` method. The approximative solution is then updated automatically using the :py:meth:`~.solve_minsetcover` function.

    :param atoms: List of objects corresponding to the atomic image regions (instances of the :py:class:`~.objects.Object` class).
    """

    def __init__(self, atoms, beta, adjacencies, try_lower_beta=DEFAULT_TRY_LOWER_BETA, lower_beta_mul=DEFAULT_LOWER_BETA_MUL):
        self.atoms = {_get_atom_label(atom): atom for atom in atoms}
        self.beta  = beta
        self.adjacencies    = adjacencies
        self.try_lower_beta = try_lower_beta
        self.lower_beta_mul = lower_beta_mul
        self. objects_by_cluster = {cluster: [atom for atom in atoms if adjacencies.get_cluster_label(_get_atom_label(atom)) == cluster] for cluster in adjacencies.cluster_labels}
        self.solution_by_cluster = {cluster: self.objects_by_cluster[cluster] for cluster in adjacencies.cluster_labels}

    def _update_partial_solution(self, cluster_label, out):
        objects = self.objects_by_cluster[cluster_label]
        partial_solution = solve_minsetcover(objects, self.beta, try_lower_beta=self.try_lower_beta, lower_beta_mul=self.lower_beta_mul, out=out)
        self.solution_by_cluster[cluster_label] = partial_solution

    def get_atom(self, atom_label):
         """Returns the object corresponding to an atomic image region.
         """
         return self.atoms[atom_label]

    def update(self, new_objects, out=None):
        """Adds new objects to the family of candidate sets :math:`\\mathscr S` and updates the solution.

        :param new_objects: An iterable of :py:class:`~.objects.Object` instances, each representing a set of atomic image regions.

        The solution of the min-weight set-cover is updated automatically.
        """
        invalidated_clusters = []
        for new_object in new_objects:
            cluster_label = self.adjacencies.get_cluster_label(list(new_object.footprint)[0])
            invalidated_clusters.append(cluster_label)
            self.objects_by_cluster[cluster_label].append(new_object)
        for cluster_label in frozenset(invalidated_clusters):
            self._update_partial_solution(cluster_label, out)

    def get_cluster_costs(self, cluster_label):
        """Returns the value of the min-weight set-cover for the subset of candidate sets which correspond to a single region of possibly clustered objects.
        """
        partial_solution = self.solution_by_cluster[cluster_label]
        return sum(c.energy for c in partial_solution) + self.beta * len(partial_solution)

    @property
    def solution(self):
        """The min-weight set-cover, i.e. the optimal family of sets of atomic image regions.

        This is the family :math:`\\mathscr X \\subseteq \\mathscr S` of objects, which is optimal with respect to the objective function

        .. math:: \\sum_{X \\in \\mathscr X} \\beta + \\nu(X)

        subject to the constraint that :math:`\\bigcup \\mathscr S = \\bigcup \\mathscr X`.
        """
        return sum((list(partial_solution) for partial_solution in self.solution_by_cluster.values()), [])

    @property
    def costs(self):
        """Returns the value of the min-weight set-cover.

        This is the optimal value of the objective function, i.e.

        .. math:: \\sum_{X \\in \\mathscr X} \\beta + \\nu(X),

        where :math:`\mathscr X` is the min-weight set-cover.
        """
        solution = self.solution
        return sum(c.energy for c in solution) + self.beta * len(solution)

from .pipeline import Stage
from ._aux import join_path, mkdir, copy_dict
from .output import get_output, Text
from .objects import compute_objects, Object
from .minsetcover import MinSetCover, DEFAULT_MAX_ITER, DEFAULT_GAMMA
from .maxsetpack import solve_maxsetpack
from .image import Image

import scipy.ndimage as ndi
import numpy as np


DEFAULT_MAX_WORK_AMOUNT = 10 ** 6


def _get_generation_log_dir(log_root_dir, generation_number):
    if log_root_dir is None: return None
    result = join_path(log_root_dir, f'gen{generation_number}')
    mkdir(result)
    return result


class PerformanceReport:
    """Reports the performance of the global energy minimization.

    :ivar direct_solution_trial_count: The number of cases in which Criterion 2 was evaluated (see the :ref:`paper <references>`).
    :ivar direct_solution_success_count: The number of cases in which Criterion 2 yielded a closed-form solution (see the :ref:`paper <references>`).
    :ivar iterative_object_count: The number of objects which would be computed if bruteforce was used instead of Algorithm 1.
    :ivar iterative_computed_object_count: The number of objects computed by Algorithm 1 (see the :ref:`paper <references>`).
    :ivar overall_object_count: The overall number of objects which would be computed if neither Algorithm 1 nor Criterion 2 was used.
    :ivar overall_computed_object_count: The overall number of computed objects.
    :ivar nontrivial_object_count: The overall number of objects which would be computed if neither Algorithm 1 nor Criterion 2 was used (except for *trivial* regions of possibly clustered objects).
    :ivar nontrivial_computed_object_count: The overall number of computed objects (except for *trivial* regions of possibly clustered objects).

    For regions of possibly clustered objects, for which the cardinality :math:`\\# U` of the universe :math:`U` is 1 or 2, always all possible objects must be computed. Since only 3 objects are possible at most (the region of possbily clustered objects either corresponds to two objects or to a single object), such regions are called *trivial*.
    """

    attributes = [
        'direct_solution_trial_count',
        'direct_solution_success_count',
        'iterative_object_count',
        'iterative_computed_object_count',
        'overall_object_count',
        'overall_computed_object_count',
        'nontrivial_object_count',
        'nontrivial_computed_object_count',
    ]
    """List of supported keyword arguments and attributes.
    """
    
    def __init__(self, **kwargs):
        for key in PerformanceReport.attributes:
            setattr(self, key, kwargs.get(key, 0))

    @property
    def direct_solution_success(self):
        """The number of cases in which Criterion 2 yielded a closed-form solution, normalized by the number of cases in which Criterion 2 was evaluated (see the :ref:`paper <references>`).
        """
        if self.direct_solution_trial_count == 0: return np.nan
        else: return self.direct_solution_success_count / self.direct_solution_trial_count
    
    @property
    def iterative_pruning_success(self):
        """The number of objects pruned by Algorithm 1, normalized by the number of objects which would be computed if bruteforce was used instead of Algorithm 1 (see the :ref:`paper <references>`).
        """
        if self.iterative_object_count == 0: return np.nan
        else: return 1 - self.iterative_computed_object_count / self.iterative_object_count
    
    @property
    def overall_pruning_success(self):
        """The number of pruned objects, normalized by the number of objects which would be computed if neither Algorithm 1 nor Criterion 2 was used (see the :ref:`paper <references>`).
        """
        if self.overall_object_count == 0: return np.nan
        else: return 1 - self.overall_computed_object_count / self.overall_object_count
    
    @property
    def nontrivial_pruning_success(self):
        """The number of pruned objects within non-trivial regions of possibly clustered objects, normalized by the number of objects in those regions which would be computed if neither Algorithm 1 nor Criterion 2 was used (see the :ref:`paper <references>`).

        This is the key performance indicator for the overall pruning performance.
        """
        if self.nontrivial_object_count == 0: return np.nan
        else: return 1 - self.nontrivial_computed_object_count / self.nontrivial_object_count
    
    def __iadd__(self, other):
        for key in PerformanceReport.attributes:
            setattr(self, key, getattr(self, key) + getattr(other, key))
        return self
    
    def _assert_integrity(self):
        assert np.isnan(   self.direct_solution_success) or (0 <= self.direct_solution_success    <= 1), f'{self.   direct_solution_success_count} / {self.direct_solution_trial_count}'
        assert np.isnan( self.iterative_pruning_success) or (0 <= self.iterative_pruning_success  <= 1), f'{self. iterative_computed_object_count} / {self.     iterative_object_count}'
        assert np.isnan(self.nontrivial_pruning_success) or (0 <= self.nontrivial_pruning_success <= 1), f'{self.nontrivial_computed_object_count} / {self.    nontrivial_object_count}'
        assert np.isnan(   self.overall_pruning_success) or (0 <= self.overall_pruning_success    <= 1), f'{self.   overall_computed_object_count} / {self.       overall_object_count}'


class GlobalEnergyMinimization(Stage):
    """Implements the global energy minimization (see :ref:`pipeline_theory_jointsegandclustersplit`).

    This stage implements Algorithm 1 and Criterion 2 of the :ref:`paper <references>`. The stage requires ``y``, ``y_mask``, ``atoms`, ``adjacencies``, ``dsm_cfg`` for input and produces ``y_img``, ``cover``, ``objects``, ``performance`` for output. Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    For Algorithm 1, there are two behaviours implemented which differ in the definition of the upper bound :math:`c_{\\text{max}}`. In *strict* mode, the original definition from the paper is used,

    .. math:: c_{\\text{max}} = c_{\\text{max}}^{\\text{strict}} = \\operatorname{MSC}(\\mathscr U) - \\sum_{u \\in U \\setminus X} \\nu(\\{u\\}),

    which guarantees that :math:`\\operatorname{MSC}(\\mathscr U_{\\# U}) = \\operatorname{MSC}(\\mathbb P(U))`. On the other hand, given a previously considered set :math:`X \\subset U` and a newly added atomic image region :math:`u \\in U`, using the upper bound

    .. math:: c_{\\text{max}} = \\min\\{ c_{\\text{max}}^{\\text{strict}}, c_{\\text{max}}^{\\text{greedy}} \\}, \\quad c_{\\text{max}}^{\\text{greedy}} = \\nu(X) + \\nu(\\{u\\}) + 2\\beta
    
    yields a more *greedy* behaviour of the algorithm, which is faster.

    Hyperparameters
    ---------------

    The following hyperparameters can be used to control this pipeline stage:

    ``global-energy-minimization/strict``
        Operates in *strict* mode if set to ``True``, and in *greedy* mode otherwise. Defaults to ``True``.

    ``global-energy-minimization/beta``
        Corresponds to the sparsity parameter :math:`\\beta` described in :ref:`pipeline_theory_jointsegandclustersplit`. Defaults to 0, or to ``AF_beta × scale^2`` if configured automatically, where ``AF_beta`` corresponds to :math:`\\beta_\\text{factor}` in the :ref:`paper <references>` and defaults to 0.66. Due to a transmission error, the values reported for ``AF_beta`` in the paper were misstated by a factor of 2 (Section 3.3, Supplemental Material 8).

    ``global-energy-minimization/max_iter``
        The number of iterations to perform for solving the *min-weight set-cover* (see :py:meth:`~superdsm.minsetcover.solve_minsetcover` and Algorithm 2 in the :ref:`paper <references>`). Iterations use an increasingly conservative merging strategy (i.e. the sparsity parameter :math:`\\beta` is reduced). Defaults to 5.

    ``global-energy-minimization/gamma``
        The factor used to reduce the sparsity parameter :math:`\\beta` after the first iteration (this is the parameter :math:`\\gamma` of Algorithm 2in the :ref:`paper <references>`, where :math:`0 < \\gamma < 1`). Defaults to 0.8.

    ``global-energy-minimization/max_seed_distance``
        Maximum distance allowed between two seed points of atomic image regions which are grouped into an image region corresponding to single object (cf. :ref:`pipeline_theory_c2freganal`). This can be used to enforce that the segmented objects will be of a maximum size, and thus to limit the computational cost by using prior knowledge. Defaults to infinity, or to ``AF_max_seed_distance × diameter`` if configured automatically (and ``AF_max_seed_distance`` defaults to infinity).

    ``global-energy-minimization/max_work_amount``
        Used to recognize a computationally intractable amount of objects due to misconfigured hyperparameters. If the number of objects *could* exceed this threshold, a ``ValueError`` is raised. The number of objects is estimated based on the structure of the adjacency graph of the atomic image regions (see :ref:`pipeline_theory_c2freganal`). Defaults to 10e6.
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(GlobalEnergyMinimization, self).__init__('global-energy-minimization',
                                                       inputs  = ['y', 'y_mask', 'atoms', 'adjacencies', 'dsm_cfg'],
                                                       outputs = ['y_img', 'cover', 'objects', 'performance'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_img             = Image.create_from_array(input_data['y'], normalize=False, mask=input_data['y_mask'])
        atoms             = input_data['atoms']
        adjacencies       = input_data['adjacencies']
        strict            = cfg.get(           'strict', True)
        beta              = cfg.get(             'beta', 0)
        max_iter          = cfg.get(         'max_iter', DEFAULT_MAX_ITER)
        gamma             = cfg.get(            'gamma', DEFAULT_GAMMA)
        max_seed_distance = cfg.get('max_seed_distance', np.inf)
        max_work_amount   = cfg.get(  'max_work_amount', DEFAULT_MAX_WORK_AMOUNT)

        assert 0 < gamma < 1

        mode = 'strict' if strict else 'fast'
        dsm_cfg = copy_dict(input_data['dsm_cfg'])
        cover, objects, performance = _compute_generations(adjacencies, y_img, atoms, log_root_dir, mode, dsm_cfg, beta, max_iter, gamma, max_seed_distance, max_work_amount, out)[2:]

        return {
            'y_img':       y_img,
            'cover':       cover,
            'objects':     objects,
            'performance': performance,
        }

    def configure_ex(self, scale, radius, diameter):
        return {
            'beta': (scale ** 2, 0.66),
            'max_seed_distance': (diameter, np.inf),
        }


def _compute_generations(adjacencies, y_img, atoms_map, log_root_dir, mode, dsm_cfg, beta=np.nan, max_iter=DEFAULT_MAX_ITER, gamma=DEFAULT_GAMMA, max_seed_distance=np.inf, max_work_amount=DEFAULT_MAX_WORK_AMOUNT, out=None):
    out = get_output(out)

    atoms = []
    for atom_label in adjacencies.atom_labels:
        c = Object()
        c.footprint = {atom_label}
        atoms.append(c)
    out.write(f'\nIteration 1:')
    compute_objects(atoms, y_img, atoms_map, dsm_cfg, _get_generation_log_dir(log_root_dir, 1), out=out)

    universes = []
    for cluster_label in adjacencies.cluster_labels:
        universe = Object()
        universe.footprint = adjacencies.get_atoms_in_cluster(cluster_label)
        universes.append(universe)
    compute_objects(universes, y_img, atoms_map, dsm_cfg, _get_generation_log_dir(log_root_dir, 0), ('Computing universe costs', 'Universe costs computed'), out=out)
    directly_solved_cluster_labels = set() ## solved using closed-form solution
    trivial_cluster_labels         = set() ## universe cardinality 1 or 2
    for cluster_label, universe in zip(adjacencies.cluster_labels, universes):
        if len(universe.footprint) <= 2: trivial_cluster_labels |= {cluster_label}
        atoms_in_cluster = [atoms[atom_label - 1] for atom_label in adjacencies.get_atoms_in_cluster(cluster_label)]
        if not all(atom.is_optimal for atom in atoms_in_cluster): continue
        atom_energies_sum = sum(atom.energy for atom in atoms_in_cluster)
        if universe.energy <= beta + atom_energies_sum:
            directly_solved_cluster_labels |= {cluster_label}

    cover = MinSetCover(atoms, beta, adjacencies, max_iter=max_iter, gamma=gamma)
    cover.update(universes, out.derive(muted=True))
    costs = [cover.costs]
    out.write(f'Solution costs: {costs[-1]:,g}')
    out.write(f'Clusters solved directly: {len(directly_solved_cluster_labels)} / {len(adjacencies.cluster_labels)}')
    performance = PerformanceReport(direct_solution_trial_count=len(adjacencies.cluster_labels), direct_solution_success_count=len(directly_solved_cluster_labels))
    
    __estimate_progress = lambda **kwargs: _estimate_progress(generations, adjacencies, max_seed_distance, max_amount=max_work_amount, skip_last=True, **kwargs)

    generations = [atoms]
    objects     =  atoms + universes
    assert len(objects) == len([frozenset(obj.footprint) for obj in objects])
    performance.nontrivial_object_count = __estimate_progress(ignored_cluster_labels=trivial_cluster_labels)[1]
    performance.   overall_object_count = performance.nontrivial_object_count + len(objects)
    performance. iterative_object_count = __estimate_progress(ignored_cluster_labels=directly_solved_cluster_labels)[1]
    performance.overall_computed_object_count = len(objects)
    if len(directly_solved_cluster_labels) < len(adjacencies.cluster_labels):

        while True:
            generation_number = 1 + len(generations)
            generation_label  = f'Iteration {generation_number}'
            out.write('')
            out.intermediate(f'{generation_label}...')

            finished_amount, remaining_amount = __estimate_progress(ignored_cluster_labels=directly_solved_cluster_labels)
            if np.isnan(finished_amount) or np.isnan(remaining_amount): progress_text = 'progress unknown'
            else:
                progress = finished_amount / (remaining_amount + finished_amount)
                progress_text = f'(finished {100 * progress:.0f}% or more)'
            out.write(f'{generation_label}: {Text.style(progress_text, Text.BOLD)}')
            
            new_generation, new_objects = _process_generation(cover, objects, generations[-1], y_img, atoms_map, adjacencies, dsm_cfg, max_seed_distance, _get_generation_log_dir(log_root_dir, generation_number), mode, directly_solved_cluster_labels, out)
            if len(new_generation) == 0: break
            generations.append(new_generation)
            objects += new_objects
            performance.nontrivial_computed_object_count += len(new_objects)
            performance.   overall_computed_object_count += len(new_objects)
            performance. iterative_computed_object_count += len(new_objects)

            cover.update(new_generation, out.derive(muted=True))
            costs.append(cover.costs)
            out.write(f'Solution costs: {costs[-1]:,g}')

    out.write('')
    out.write(f'Non-trivial pruning: {100 * performance.nontrivial_pruning_success:.1f}% (computed {performance.nontrivial_computed_object_count} / {performance.nontrivial_object_count})')
    performance._assert_integrity()
    return generations, costs, cover, objects, performance


def _get_max_distance(footprint, new_atom_label, adjacencies):
    """Computes the maximum distance between the seed of `new_atom_label` and a seed point in `footprint`
    """
    assert new_atom_label not in footprint
    maximum_distance = 0
    new_atom_seed = adjacencies.get_seed(new_atom_label)
    for label in footprint:
        distance = np.linalg.norm(adjacencies.get_seed(label) - new_atom_seed)
        maximum_distance = max((maximum_distance, distance))
    return maximum_distance


def _is_within_max_seed_distance(footprint, new_atom_label, adjacencies, max_seed_distance):
    if np.isinf(max_seed_distance): return True
    maximum_distance = _get_max_distance(footprint, new_atom_label, adjacencies)
    return maximum_distance <= max_seed_distance


def _iterate_generation(previous_generation, adjacencies, max_seed_distance, get_footprint=lambda item: item, ignored_cluster_labels=set(), skip_last=False):
    existing_footprints = set()
    for item in previous_generation:
        footprint = get_footprint(item)
        cluster_label = adjacencies.get_cluster_label(list(footprint)[0])
        if cluster_label in ignored_cluster_labels: continue
        if skip_last and len(footprint) + 1 == len(adjacencies.get_atoms_in_cluster(cluster_label)): continue
        adjacent_atoms = set()
        for atom in footprint:
            adjacent_atoms |= adjacencies[atom] - footprint
        for new_atom_label in adjacent_atoms:
            if not _is_within_max_seed_distance(footprint, new_atom_label, adjacencies, max_seed_distance): continue
            new_footprint = frozenset(footprint | {new_atom_label})
            if new_footprint not in existing_footprints:
                existing_footprints |= {new_footprint}
                yield item, new_footprint, new_atom_label


def _get_next_generation(previous_generation, adjacencies, max_seed_distance, **kwargs):
    return [new_footprint for _, new_footprint, _ in _iterate_generation(previous_generation, adjacencies, max_seed_distance, **kwargs)]


def _estimate_progress(generations, adjacencies, max_seed_distance, max_amount=DEFAULT_MAX_WORK_AMOUNT, ignored_cluster_labels=set(), skip_last=False):
    previous_generation = [c.footprint for c in generations[-1]]
    remaining_amount    =  0
    while len(previous_generation) > 0:
        next_generation     = _get_next_generation(previous_generation, adjacencies, max_seed_distance, ignored_cluster_labels=ignored_cluster_labels, skip_last=skip_last)
        remaining_amount   += len(next_generation)
        previous_generation = next_generation
        if remaining_amount > max_amount: raise ValueError('estimated work amount is too large')
    finished_amount = len(sum(generations, []))
    return finished_amount, remaining_amount


def _process_generation(cover, objects, previous_generation, y, atoms_map, adjacencies, dsm_cfg, max_seed_distance, log_root_dir, mode, ignored_cluster_labels, out):
    new_objects = []
    new_objects_energy_thresholds = []
    discarded = 0
    current_cluster_label = None
    for object, new_object_footprint, new_atom_label in _iterate_generation(previous_generation, adjacencies, max_seed_distance, lambda c: c.footprint, ignored_cluster_labels, skip_last=True):
        cluster_label = adjacencies.get_cluster_label(list(object.footprint)[0])
        if current_cluster_label != cluster_label:
            current_cluster_label = cluster_label
            current_cluster_costs = cover.get_cluster_costs(cluster_label)

        new_object = Object()
        new_object.footprint = new_object_footprint

        remaining_atoms = adjacencies.get_atoms_in_cluster(cluster_label) - new_object_footprint
        min_remaining_atom_costs = sum(cover.get_atom(atom_label).energy for atom_label in remaining_atoms)
        new_object_maxsetpack = sum(c.energy for c in solve_maxsetpack([c for c in objects if c.is_optimal and c.footprint.issubset(new_object.footprint)], out=out.derive(muted=True)))
        min_new_object_costs = cover.beta + max((object.energy + cover.get_atom(new_atom_label).energy, new_object_maxsetpack))
        max_new_object_costs = current_cluster_costs - min_remaining_atom_costs
        if mode == 'strict':
            pass
        elif mode == 'fast':
            max_new_object_costs = min((max_new_object_costs, object.energy + cover.get_atom(new_atom_label).energy + 2 * cover.beta))
        else:
            raise ValueError(f'Unknown mode "{mode}"')
        if max_new_object_costs < min_new_object_costs:
            discarded += 1
        else:
            new_objects_energy_thresholds.append(max_new_object_costs - cover.beta)
            new_objects.append(new_object)

    compute_objects(new_objects, y, atoms_map, dsm_cfg, log_root_dir, out=out)

    next_generation = []
    for new_object_idx, new_object in enumerate(new_objects):
        if new_object.energy < new_objects_energy_thresholds[new_object_idx]:
            next_generation.append(new_object)
        else:
            discarded += 1
            new_object.fg_fragment = None ## save memory, we will only only need the footprint and the energy of the object
        new_object.cidx = new_object_idx ## for debugging purposes
    out.write(f'Next iteration: {len(next_generation)} ({discarded} discarded, {mode} mode)')
    return next_generation, new_objects

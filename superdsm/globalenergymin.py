from .pipeline import Stage
from ._aux import join_path, mkdir, get_discarded_workload, copy_dict
from .output import get_output, Text
from .objects import compute_objects, Object
from .minsetcover import MinSetCover, DEFAULT_TRY_LOWER_BETA, DEFAULT_LOWER_BETA_MUL, DEFAULT_TRY_LOWER_BETA, DEFAULT_LOWER_BETA_MUL
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


class GlobalEnergyMinimization(Stage):
    """Implements the global energy minimization (see :ref:`pipeline_theory_jointsegandclustersplit`).

    This stage implements Algorithm 1 and Criterion 2 of the :ref:`paper <references>`. The stage requires ``y``, ``y_mask``, ``atoms`, ``adjacencies``, ``dsm_cfg`` for input and produces ``y_img``, ``cover``, ``objects``, ``workload`` for output. Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    Hyperparameters
    ---------------

    The following hyperparameters can be used to control this pipeline stage:

    ``global-energy-minimization/conservative``
        tbd. Defaults to ``True``.

    ``global-energy-minimization/beta``
        Corresponds to the constant term :math:`\\beta` described in :ref:`pipeline_theory_jointsegandclustersplit`. Defaults to 0, or to ``AF_beta × scale^2`` if configured automatically, where ``AF_beta`` corresponds to :math:`\\beta_\\text{factor}` in the :ref:`paper <references>` and defaults to 0.66. Due to a transmission error, the values reported for ``AF_beta`` in the paper were misstated by a factor of 2 (Section 3.3, Supplemental Material 8).

    ``global-energy-minimization/try_lower_beta``
        tbd.

    ``global-energy-minimization/lower_beta_mul``
        tbd.

    ``global-energy-minimization/max_seed_distance``
        Maximum distance allowed between two seed points of atomic image regions which are grouped into an image region corresponding to single object (cf. :ref:`pipeline_theory_c2freganal`). This can be used to enforce that the segmented objects will be of a maximum size, and thus to limit the computational cost by using prior knowledge. Defaults to infinity, or to ``AF_max_seed_distance × diameter`` if configured automatically (and ``AF_max_seed_distance`` defaults to infinity).

    ``global-energy-minimization/max_work_amount``
        tbd.
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(GlobalEnergyMinimization, self).__init__('global-energy-minimization',
                                                       inputs  = ['y', 'y_mask', 'atoms', 'adjacencies', 'dsm_cfg'],
                                                       outputs = ['y_img', 'cover', 'objects', 'workload'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_img             = Image.create_from_array(input_data['y'], normalize=False, mask=input_data['y_mask'])
        atoms             = input_data['atoms']
        adjacencies       = input_data['adjacencies']
        conservative      = cfg.get(     'conservative', True)
        beta              = cfg.get(             'beta', 0)
        try_lower_beta    = cfg.get(   'try_lower_beta', DEFAULT_TRY_LOWER_BETA)
        lower_beta_mul    = cfg.get(   'lower_beta_mul', DEFAULT_LOWER_BETA_MUL)
        max_seed_distance = cfg.get('max_seed_distance', np.inf)
        max_work_amount   = cfg.get(  'max_work_amount', DEFAULT_MAX_WORK_AMOUNT)

        assert 0 < lower_beta_mul < 1

        mode  = 'conservative' if conservative else 'fast'
        dsm_cfg = copy_dict(input_data['dsm_cfg'])
        cover, objects, workload = _compute_generations(adjacencies, y_img, atoms, log_root_dir, mode, dsm_cfg, beta, try_lower_beta, lower_beta_mul, max_seed_distance, max_work_amount, out)[2:]

        return {
            'y_img':    y_img,
            'cover':    cover,
            'objects':  objects,
            'workload': workload,
        }

    def configure_ex(self, scale, radius, diameter):
        return {
            'beta': (scale ** 2, 0.66),
            'max_seed_distance': (diameter, np.inf),
        }


def _compute_generations(adjacencies, y_img, atoms_map, log_root_dir, mode, dsm_cfg, beta=np.nan, try_lower_beta=DEFAULT_TRY_LOWER_BETA, lower_beta_mul=DEFAULT_LOWER_BETA_MUL, max_seed_distance=np.inf, max_work_amount=DEFAULT_MAX_WORK_AMOUNT, out=None):
    assert mode != 'bruteforce', 'mode "bruteforce" not supported anymore'
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
    trivial_cluster_labels = set()
    for cluster_label, universe in zip(adjacencies.cluster_labels, universes):
        atoms_in_cluster = [atoms[atom_label - 1] for atom_label in adjacencies.get_atoms_in_cluster(cluster_label)]
        if not all(atom.is_optimal for atom in atoms_in_cluster): continue
        atom_energies_sum = sum(atom.energy for atom in atoms_in_cluster)
        if universe.energy <= beta + atom_energies_sum:
            trivial_cluster_labels |= {cluster_label}

    cover = MinSetCover(atoms, beta, adjacencies, try_lower_beta, lower_beta_mul)
    cover.update(universes, out.derive(muted=True))
    costs = [cover.costs]
    out.write(f'Solution costs: {costs[-1]:,g}')
    out.write(f'Clusters solved trivially: {len(trivial_cluster_labels)} / {len(adjacencies.cluster_labels)}')
    
    if max_work_amount is None:
        __estimate_progress = lambda *args, **kwargs: (np.nan, np.nan)
    else:
        __estimate_progress = lambda *args, **kwargs: _estimate_progress(*args, max_amount=max_work_amount, **kwargs)

    generations = [atoms]
    objects     =  atoms + universes
    total_workload = len(objects) + __estimate_progress(generations, adjacencies, max_seed_distance, skip_last=False)[1]
    if len(trivial_cluster_labels) < len(adjacencies.cluster_labels):

        while True:
            generation_number = 1 + len(generations)
            generation_label  = f'Iteration {generation_number}'
            out.write('')
            out.intermediate(f'{generation_label}...')

            finished_amount, remaining_amount = __estimate_progress(generations, adjacencies, max_seed_distance, ignored_cluster_labels=trivial_cluster_labels, skip_last=True)
            if np.isnan(finished_amount) or np.isnan(remaining_amount): progress_text = 'progress unknown'
            else:
                progress = finished_amount / (remaining_amount + finished_amount)
                progress_text = f'(finished {100 * progress:.0f}% or more)'
            out.write(f'{generation_label}: {Text.style(progress_text, Text.BOLD)}')
            
            new_generation, new_objects = _process_generation(cover, objects, generations[-1], y_img, atoms_map, adjacencies, dsm_cfg, max_seed_distance, _get_generation_log_dir(log_root_dir, generation_number), mode, trivial_cluster_labels, out)
            if len(new_generation) == 0: break
            generations.append(new_generation)
            objects += new_objects

            cover.update(new_generation, out.derive(muted=True))
            costs.append(cover.costs)
            out.write(f'Solution costs: {costs[-1]:,g}')

    if np.isnan(total_workload): discarded_workload = np.nan
    else: discarded_workload = get_discarded_workload(len(objects), total_workload)
    out.write('')
    out.write(f'Discarded workload: {100 * discarded_workload:.1f}%')
    return generations, costs, cover, objects, total_workload


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
    new_objects_thresholds = []
    discarded = 0
    current_cluster_label = None
    for object, new_object_footprint, new_atom_label in _iterate_generation(previous_generation, adjacencies, max_seed_distance, lambda c: c.footprint, ignored_cluster_labels, skip_last=True):
        cluster_label = adjacencies.get_cluster_label(list(object.footprint)[0])
        if current_cluster_label != cluster_label:
            current_cluster_label = cluster_label
            current_cluster_costs = cover.get_cluster_costs(cluster_label) if mode != 'bruteforce' else np.inf

        new_object = Object()
        new_object.footprint = new_object_footprint

        remaining_atoms = adjacencies.get_atoms_in_cluster(cluster_label) - new_object_footprint
        min_remaining_atom_costs = sum(cover.get_atom(atom_label).energy for atom_label in remaining_atoms)
        if mode == 'conservative':
            max_new_object_energy = current_cluster_costs - cover.beta - min_remaining_atom_costs
        elif mode == 'fast':
            max_new_object_energy = object.energy + cover.get_atom(new_atom_label).energy + cover.beta
        else:
            raise ValueError(f'unknown mode "{mode}"')
        new_object_maxsetpack = sum(c.energy for c in solve_maxsetpack([c for c in objects if c.is_optimal and c.footprint.issubset(new_object.footprint)], out=out.derive(muted=True)))
        min_new_object_energy = max((object.energy + cover.get_atom(new_atom_label).energy, new_object_maxsetpack))
        if max_new_object_energy < min_new_object_energy:
            discarded += 1
        else:
            new_objects_thresholds.append(max_new_object_energy)
            new_objects.append(new_object)

    compute_objects(new_objects, y, atoms_map, dsm_cfg, log_root_dir, out=out)

    next_generation = []
    for new_object_idx, new_object in enumerate(new_objects):
        if new_object.energy < new_objects_thresholds[new_object_idx]:
            next_generation.append(new_object)
        else:
            discarded += 1
            new_object.fg_fragment = None ## save memory, we will only only need the footprint and the energy of the object
        new_object.cidx = new_object_idx ## for debugging purposes
    out.write(f'Next generation: {len(next_generation)} (discarded: {discarded})')
    return next_generation, new_objects

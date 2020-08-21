import gocell.config
import gocell.pipeline
import gocell.aux
import gocell.candidates
import gocell.minsetcover
import gocell.maxsetpack

import scipy.ndimage as ndi
import numpy as np


MODELFIT_KWARGS_DEFAULTS = {
    'cachesize': 1,
    'sparsity_tol': 0,
    'init': 'gocell',
    'smooth_amount': 10,
    'epsilon': 0.1,
    'rho': 0.8,
    'scale': 1000,
    'smooth_subsample': 20,
    'gaussian_shape_multiplier': 2,
    'smooth_mat_dtype': 'float32',
    'min_background_margin': 20,
    'max_background_margin': 100
}


def _get_generation_log_dir(log_root_dir, generation_number):
    if log_root_dir is None: return None
    result = gocell.aux.join_path(log_root_dir, f'gen{generation_number}')
    gocell.aux.mkdir(result)
    return result


class GenerationStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(GenerationStage, self).__init__('generations',
                                              inputs  = ['y', 'y_mask', 'g_atoms', 'adjacencies'],
                                              outputs = ['y_surface', 'cover', 'candidates', 'workload'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_surface         = gocell.surface.Surface.create_from_image(input_data['y'], normalize=False, mask=input_data['y_mask'])
        g_atoms           = input_data['g_atoms']
        adjacencies       = input_data['adjacencies']
        conservative      = gocell.config.get_value(cfg,      'conservative', True)
        alpha             = gocell.config.get_value(cfg,             'alpha', 0)
        try_lower_alpha   = gocell.config.get_value(cfg,   'try_lower_alpha', gocell.minsetcover.DEFAULT_TRY_LOWER_ALPHA)
        lower_alpha_mul   = gocell.config.get_value(cfg,   'lower_alpha_mul', gocell.minsetcover.DEFAULT_LOWER_ALPHA_MUL)
        max_seed_distance = gocell.config.get_value(cfg, 'max_seed_distance', np.inf)

        assert 0 < lower_alpha_mul < 1

        mode = 'conservative' if conservative else 'fast'
        cover, candidates, workload = compute_generations(adjacencies, y_surface, g_atoms, log_root_dir, mode, cfg, alpha, try_lower_alpha, lower_alpha_mul, max_seed_distance, out)[2:]

        return {
            'y_surface':  y_surface,
            'cover':      cover,
            'candidates': candidates,
            'workload':   workload
        }


def compute_generations(adjacencies, y_surface, g_atoms, log_root_dir, mode, cfg, alpha=np.nan, try_lower_alpha=gocell.minsetcover.DEFAULT_TRY_LOWER_ALPHA, lower_alpha_mul=gocell.minsetcover.DEFAULT_LOWER_ALPHA_MUL, max_seed_distance=np.inf, out=None):
    assert mode != 'bruteforce', 'mode "bruteforce" not supported anymore'
    out = gocell.aux.get_output(out)

    modelfit_kwargs = {
        key: gocell.config.get_value(cfg, key, MODELFIT_KWARGS_DEFAULTS[key]) for key in MODELFIT_KWARGS_DEFAULTS.keys()
    }

    atoms = []
    for atom_label in adjacencies.atom_labels:
        c = gocell.candidates.Candidate()
        c.footprint = {atom_label}
        atoms.append(c)
    out.write(f'\nGeneration 1:')
    gocell.candidates.process_candidates(atoms, y_surface, g_atoms, modelfit_kwargs, _get_generation_log_dir(log_root_dir, 1), out=out)

    universes = []
    for cluster_label in adjacencies.cluster_labels:
        universe = gocell.candidates.Candidate()
        universe.footprint = adjacencies.get_atoms_in_cluster(cluster_label)
        universes.append(universe)
    gocell.candidates.process_candidates(universes, y_surface, g_atoms, modelfit_kwargs, _get_generation_log_dir(log_root_dir, 0), ('Computing universe costs', 'Universe costs computed'), out=out)
    trivial_cluster_labels = set()
    for cluster_label, universe in zip(adjacencies.cluster_labels, universes):
        atom_energies_sum = sum(atoms[atom_label - 1].energy for atom_label in adjacencies.get_atoms_in_cluster(cluster_label))
        if universe.energy <= alpha + atom_energies_sum:
            trivial_cluster_labels |= {cluster_label}

    cover = gocell.minsetcover.MinSetCover(atoms, alpha, adjacencies, try_lower_alpha, lower_alpha_mul)
    cover.update(universes, out.derive(muted=True))
    costs = [cover.costs]
    out.write(f'Solution costs: {costs[-1]:,g}')
    out.write(f'Clusters solved trivially: {len(trivial_cluster_labels)} / {len(adjacencies.cluster_labels)}')

    generations    = [atoms]
    candidates     =  atoms + universes
    total_workload = len(candidates) + _estimate_progress(generations, adjacencies, max_seed_distance, skip_last=False)[1]
    if len(trivial_cluster_labels) < len(adjacencies.cluster_labels):

        while True:
            generation_number = 1 + len(generations)
            generation_label  = f'Generation {generation_number}'
            out.write('')
            out.intermediate(f'{generation_label}...')

            finished_amount, remaining_amount = _estimate_progress(generations, adjacencies, max_seed_distance, ignored_cluster_labels=trivial_cluster_labels, skip_last=True)
            progress = finished_amount / (remaining_amount + finished_amount)
            progress_text = '** WARNING ** COMPUTATIONAL LOAD TOO HIGH **' if progress is None else f'(finished {100 * progress:.0f}% or more)'
            out.write(f'{generation_label}: {gocell.aux.Text.style(progress_text, gocell.aux.Text.BOLD)}')
            
            new_generation, new_candidates = _process_generation(cover, candidates, generations[-1], y_surface, g_atoms, adjacencies, modelfit_kwargs, max_seed_distance, _get_generation_log_dir(log_root_dir, generation_number), mode, trivial_cluster_labels, out)
            if len(new_generation) == 0: break
            generations.append(new_generation)
            candidates += new_candidates

            cover.update(new_generation, out.derive(muted=True))
            costs.append(cover.costs)
            out.write(f'Solution costs: {costs[-1]:,g}')

    discarded_workload = gocell.aux.get_discarded_workload(len(candidates), total_workload)
    out.write('')
    out.write(f'Discarded workload: {100 * discarded_workload:.1f}%')
    return generations, costs, cover, candidates, total_workload


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


def _estimate_progress(generations, adjacencies, max_seed_distance, max_amount=10**6, ignored_cluster_labels=set(), skip_last=False):
    previous_generation = [c.footprint for c in generations[-1]]
    remaining_amount    =  0
    while len(previous_generation) > 0:
        next_generation     = _get_next_generation(previous_generation, adjacencies, max_seed_distance, ignored_cluster_labels=ignored_cluster_labels, skip_last=skip_last)
        remaining_amount   += len(next_generation)
        previous_generation = next_generation
        if remaining_amount > max_amount: return None
    finished_amount = len(sum(generations, []))
    return finished_amount, remaining_amount


def _process_generation(cover, candidates, previous_generation, y, g_atoms, adjacencies, modelfit_kwargs, max_seed_distance, log_root_dir, mode, ignored_cluster_labels, out):
    new_candidates = []
    new_candidate_thresholds = []
    discarded = 0
    candidates_by_cluster = {cluster_label: [c for c in candidates if adjacencies.get_cluster_label(list(c.footprint)[0]) == cluster_label] for cluster_label in adjacencies.cluster_labels}
    current_cluster_label = None
    for candidate, new_candidate_footprint, new_atom_label in _iterate_generation(previous_generation, adjacencies, max_seed_distance, lambda c: c.footprint, ignored_cluster_labels, skip_last=True):
        cluster_label = adjacencies.get_cluster_label(list(candidate.footprint)[0])
        if current_cluster_label != cluster_label:
            current_cluster_label = cluster_label
            current_cluster_costs = cover.get_cluster_costs(cluster_label) if mode != 'bruteforce' else np.inf

        new_candidate = gocell.candidates.Candidate()
        new_candidate.footprint = new_candidate_footprint

        remaining_atoms = adjacencies.get_atoms_in_cluster(cluster_label) - new_candidate_footprint
        min_remaining_atom_costs = sum(cover.get_atom(atom_label).energy for atom_label in remaining_atoms)
        if mode == 'conservative':
            max_new_candidate_energy = current_cluster_costs - cover.alpha - min_remaining_atom_costs
        elif mode == 'fast':
            max_new_candidate_energy = candidate.energy + cover.get_atom(new_atom_label).energy + cover.alpha
        else:
            raise ValueError(f'unknown mode "{mode}"')
        new_candidate_maxsetpack = sum(c.energy for c in gocell.maxsetpack.solve_maxsetpack([c for c in candidates if c.is_optimal and c.footprint.issubset(new_candidate.footprint)], out=out.derive(muted=True)))
        min_new_candidate_energy = max((candidate.energy + cover.get_atom(new_atom_label).energy, new_candidate_maxsetpack))
        if max_new_candidate_energy < min_new_candidate_energy:
            discarded += 1
        else:
            new_candidate_thresholds.append(max_new_candidate_energy)
            new_candidates.append(new_candidate)

    gocell.candidates.process_candidates(new_candidates, y, g_atoms, modelfit_kwargs, log_root_dir, out=out)

    next_generation = []
    for new_candidate_idx, new_candidate in enumerate(new_candidates):
        if new_candidate.energy < new_candidate_thresholds[new_candidate_idx]:
            next_generation.append(new_candidate)
        else:
            discarded += 1
            new_candidate.fg_fragment = None ## save memory, we will only only need the footprint and the energy of the candidate
        new_candidate.cidx = new_candidate_idx ## for debugging purposes
    out.write(f'Next generation: {len(next_generation)} (discarded: {discarded})')
    return next_generation, new_candidates


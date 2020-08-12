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
                                              outputs = ['y_surface', 'generations', 'cover', 'costs', 'candidates'])

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
        generations, costs, cover, candidates = compute_generations(adjacencies, y_surface, g_atoms, log_root_dir, mode, cfg, alpha, try_lower_alpha, lower_alpha_mul, max_seed_distance, out)

        return {
            'y_surface':   y_surface,
            'generations': generations,
            'costs':       costs,
            'cover':       cover,
            'candidates':  candidates
        }


def compute_generations(adjacencies, y_surface, g_atoms, log_root_dir, mode, cfg, alpha=np.nan, try_lower_alpha=gocell.minsetcover.DEFAULT_TRY_LOWER_ALPHA, lower_alpha_mul=gocell.minsetcover.DEFAULT_LOWER_ALPHA_MUL, max_seed_distance=np.inf, out=None):
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
    gocell.candidates.process_candidates(atoms, y_surface, g_atoms, modelfit_kwargs, _get_generation_log_dir(log_root_dir, 1), out)

    if mode == 'bruteforce':
        cover = None
        costs = None
    else:
        cover = gocell.minsetcover.MinSetCover(atoms, alpha, adjacencies, try_lower_alpha, lower_alpha_mul)
        costs = [cover.costs]
        out.write(f'Solution costs: {costs[-1]:,g}')

    generations = [atoms]
    candidates  = sum(generations, [])
    while True:
        generation_number = 1 + len(generations)
        out.write(f'\nGeneration {generation_number}:')
        
        new_generation, new_candidates = _iterate_generation(cover, candidates, generations[-1], y_surface, g_atoms, adjacencies, modelfit_kwargs, max_seed_distance, _get_generation_log_dir(log_root_dir, generation_number), mode, out)
        if len(new_generation) == 0: break
        generations.append(new_generation)
        candidates += new_candidates

        if mode != 'bruteforce':
            cover.update(new_generation, out.derive(muted=True))
            costs.append(cover.costs)
            out.write(f'Solution costs: {costs[-1]:,g}')

    return generations, costs, cover, candidates


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


def _iterate_generation(cover, candidates, previous_generation, y, g_atoms, adjacencies, modelfit_kwargs, max_seed_distance, log_root_dir, mode, out):
    new_candidates = []
    new_candidate_thresholds = []
    discarded = 0
    existing_candidate_footprints = set()
    candidates_by_cluster = {cluster_label: [c for c in candidates if adjacencies.get_cluster_label(list(c.footprint)[0]) == cluster_label] for cluster_label in adjacencies.cluster_labels}
    for cidx, candidate in enumerate(previous_generation):
        adjacent_atoms = set()
        for atom in candidate.footprint:
            adjacent_atoms |= adjacencies[atom] - candidate.footprint
            
        cluster_label = adjacencies.get_cluster_label(list(candidate.footprint)[0])
        current_cluster_costs = cover.get_cluster_costs(cluster_label) if mode != 'bruteforce' else np.inf
        
        for new_atom_label in adjacent_atoms:
            if not _is_within_max_seed_distance(candidate.footprint, new_atom_label, adjacencies, max_seed_distance): continue
            new_candidate = gocell.candidates.Candidate()
            new_candidate.footprint = candidate.footprint | {new_atom_label}
            new_candidate_footprint = frozenset(new_candidate.footprint)
            if new_candidate_footprint not in existing_candidate_footprints:
                existing_candidate_footprints |= {new_candidate_footprint}

                if mode == 'bruteforce':
                    new_candidates.append(new_candidate)
                else:
                    remaining_atoms = adjacencies.get_atoms_in_cluster(cluster_label) - new_candidate_footprint
                    min_remaining_atom_costs = sum(cover.get_atom(atom_label).energy for atom_label in remaining_atoms)
                    min_remaining_costs = min((min_remaining_atom_costs + cover.alpha, sum(c.energy for c in gocell.maxsetpack.solve_maxsetpack([c for c in candidates_by_cluster[cluster_label] if len(c.footprint & new_candidate.footprint) == 0], out=out.derive(muted=True)))))
                    if mode == 'conservative':
                        max_new_candidate_energy = current_cluster_costs - cover.alpha - min_remaining_atom_costs
                    elif mode == 'fast':
                        max_new_candidate_energy = candidate.energy + cover.get_atom(new_atom_label).energy + cover.alpha
                    else:
                        raise ValueError(f'unknown mode "{mode}"')
                    new_candidate_maxsetpack = sum(c.energy for c in gocell.maxsetpack.solve_maxsetpack([c for c in candidates if c.footprint.issubset(new_candidate.footprint)], out=out.derive(muted=True)))
                    min_new_candidate_energy = max((candidate.energy + cover.get_atom(new_atom_label).energy, new_candidate_maxsetpack))
                    if max_new_candidate_energy < min_new_candidate_energy:
                        discarded += 1
                    else:
                        new_candidate_thresholds.append(max_new_candidate_energy)
                        new_candidates.append(new_candidate)

    gocell.candidates.process_candidates(new_candidates, y, g_atoms, modelfit_kwargs, log_root_dir, out=out)

    next_generation = []
    for new_candidate_idx, new_candidate in enumerate(new_candidates):
        if mode == 'bruteforce' or new_candidate.energy < new_candidate_thresholds[new_candidate_idx]:
            next_generation.append(new_candidate)
        else:
            discarded += 1
            new_candidate.fg_fragment = None ## save memory, we will only only need the footprint and the energy of the candidate
        new_candidate.cidx = new_candidate_idx ## for debugging purposes
    out.write(f'Next generation: {len(next_generation)} (discarded: {discarded})')
    return next_generation, new_candidates


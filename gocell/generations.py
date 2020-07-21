import gocell.config
import gocell.pipeline
import gocell.aux
import gocell.candidates
import gocell.minsetcover

import scipy.ndimage as ndi
import numpy as np


MODELFIT_KWARGS_DEFAULTS = {
    'cachesize': 1,
    'sparsity_tol': 1e-8,
    'init': 'gocell',
    'smooth_amount': 10,
    'epsilon': 0.1,
    'rho': 0.8,
    'scale': 1000,
    'smooth_subsample': 20,
    'gaussian_shape_multiplier': 2,
    'smooth_mat_max_allocations': np.inf,
    'smooth_mat_dtype': 'float32'
}


class GenerationStage(gocell.pipeline.Stage):

    def __init__(self):
        super(SeedStage, self).__init__('generations',
                                        inputs  = ['y', 'g_atoms', 'adjacencies'],
                                        outputs = ['y_surface', 'generations', 'cover', 'costs'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_surface    = gocell.surface.create_from_image(input_data['y'], normalize=False)
        g_atoms      = input_data['g_atoms']
        adjacencies  = input_data['adjacencies']
        alpha        = gocell.config.get_value(cfg,        'alpha',    0)
        conservative = gocell.config.get_value(cfg, 'conservative', True)

        modelfit_kwargs = {key: gocell.config.get_value(cfg, key, MODELFIT_KWARGS_DEFAULTS[key] for key in MODELFITK_WARGS_DEFAULTS.keys()}

        atoms = []
        for atom_label in adjacencies.atom_labels:
            c = gocell.candidates.Candidate()
            c.footprint = {atom_label}
            atoms.append(c)
        out.write(f'\nGeneration 1:')
        process_candidates(atoms, y_surface, g_atoms, modelfit_kwargs, out=out.derive())

        cover = minsetcover.MinSetCover(atoms, alpha, adjacencies)
        costs = [cover.costs]
        out.write(f'Solution costs: {costs[-1]:g}')

        generations = [atoms]
        while True:
            out.write(f'\nGeneration {1 + len(generations)}:')
            
            new_generation = _iterate_generation(cover, generations[-1], y_surface, g_atoms, adjacencies, modelfit_kwargs, conservative, out)
            if len(new_generation) == 0: break

            generations.append(new_generation)
            cover.update(new_generation, out)
            costs.append(cover.costs)
            out.write(f'Solution costs: {costs[-1]:g}')

        return {
            'y_surface': y_surface,
            'generations': generations
            'cover': cover,
            'costs': costs
        }


def _iterate_generation(cover, previous_generation, y, g_atoms, adjacencies, modelfit_kwargs, conservative, out):
    out = gocell.aux.get_output(out)
    next_generation = []
    new_candidates  = []
    new_candidate_thresholds = []
    discarded = 0
    existing_candidate_footprints = set()
    for cidx, candidate in enumerate(previous_generation):
        adjacent_atoms = set()
        for atom in candidate.footprint:
            adjacent_atoms |= adjacencies[atom] - candidate.footprint
            
        cluster_label = adjacencies.get_cluster_label(list(candidate.footprint)[0])
        max_cluster_costs = cover.get_cluster_costs(cluster_label)
        
        for new_atom_label in adjacent_atoms:
            new_candidate = gocell.candidates.Candidate()
            new_candidate.footprint = candidate.footprint | {new_atom_label}
            new_candidate_footprint = frozenset(new_candidate.footprint)
            if new_candidate_footprint not in existing_candidate_footprints:
                
                remaining_atoms = adjacencies.get_superpixels_in_cluster(cluster_label) - new_candidate_footprint
                min_remaining_atom_costs = sum(cover.get_atom(atom_label).energy for atom_label in remaining_atoms)
                if conservative:
                    max_new_candidate_energy = max_cluster_costs - cover.alpha - min_remaining_atom_costs
                else:
                    max_new_candidate_energy = candidate.energy + cover.get_atom(new_atom_label).energy + cover.alpha
                if max_new_candidate_energy < candidate.energy + cover.get_atom(new_atom_label).energy:
                    discarded += 1
                else:
                    new_candidate_thresholds.append(max_new_candidate_energy)
                    new_candidates.append(new_candidate)
                existing_candidate_footprints |= {new_candidate_footprint}
    gocell.candidates.process_candidates(new_candidates, y, g_atoms, modelfit_kwargs, out=out)
    for new_candidate_idx, new_candidate in enumerate(new_candidates):
        if new_candidate.energy < new_candidate_thresholds[new_candidate_idx]:
            next_generation.append(new_candidate)
        else:
            discarded += 1
    out.write(f'New candidates: {len(next_generation)} (discarded: {discarded})')
    return next_generation


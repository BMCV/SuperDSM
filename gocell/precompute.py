import gocell.config
import gocell.pipeline
import gocell.generations
import gocell.minsetcover

import scipy.ndimage as ndi
import numpy as np


class PrecomputeStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = False

    def __init__(self):
        super(PrecomputeStage, self).__init__('precompute',
                                              inputs  = ['y', 'g_atoms', 'adjacencies'],
                                              outputs = ['y_surface', 'precomputed_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_surface    = gocell.surface.Surface.create_from_image(input_data['y'], normalize=False)
        g_atoms      = input_data['g_atoms']
        adjacencies  = input_data['adjacencies']

        generations, costs, cover = gocell.generations.compute_generations(adjacencies, y_surface, g_atoms, log_root_dir, 'bruteforce', cfg, out=out)

        return {
            'y_surface': y_surface,
            'precomputed_candidates': precomputed_candidates
        }


class MinSetCoverStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = False

    def __init__(self):
        super(MinSetCoverStage, self).__init__('minsetcover',
                                               inputs  = ['y_surface', 'adjacencies', 'precomputed_candidates'],
                                               outputs = ['cover'])

    def process(self, input_data, cfg, out, log_root_dir):
        y_surface   = input_data['y_surface']
        adjacencies = input_data['adjacencies']
        candidates  = input_data['precomputed_candidates']
        alpha       = gocell.config.get_value(cfg, 'alpha', 0)

        atoms = []
        for atom_label in adjacencies.atom_labels:
            c = gocell.candidates.Candidate()
            c.footprint = {atom_label}
            atoms.append(c)

        cover = gocell.minsetcover.MinSetCover(atoms, alpha, adjacencies)
        cover.update(candidates, out=out)

        out.write(f'\nSolution costs: {cover.costs[-1]:,g}')

        return {
            'cover': cover
        }


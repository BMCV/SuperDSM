import gocell.pipeline
import gocell.config
import gocell.candidates

import scipy.ndimage      as ndi
import skimage.morphology as morph

import ray
import math, os
import numpy as np


class Postprocessing(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['cover', 'y_surface', 'g_atoms', 'g_raw'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        max_energy_rate         = gocell.config.get_value(cfg,         'max_energy_rate',  np.inf)
        discard_image_boundary  = gocell.config.get_value(cfg,  'discard_image_boundary',   False)
        min_boundary_obj_radius = gocell.config.get_value(cfg, 'min_boundary_obj_radius',       0)
        min_obj_radius          = gocell.config.get_value(cfg,          'min_obj_radius',       0)
        max_obj_radius          = gocell.config.get_value(cfg,          'max_obj_radius',  np.inf)
        mask_stdamp             = gocell.config.get_value(cfg,             'mask_stdamp',       2)
        mask_max_distance       = gocell.config.get_value(cfg,       'mask_max_distance',       0)
        mask_smoothness         = gocell.config.get_value(cfg,         'mask_smoothness',       3)
        exterior_scale          = gocell.config.get_value(cfg,          'exterior_scale',       5)
        exterior_offset         = gocell.config.get_value(cfg,         'exterior_offset',       5)
        min_contrast_response   = gocell.config.get_value(cfg,   'min_contrast_response', -np.inf)

        params = {
            'y':                 input_data['y_surface'],
            'g':                 input_data['g_raw'],
            'g_atoms':           input_data['g_atoms'],
            'g_smooth':          ndi.gaussian_filter(input_data['g_raw'], mask_smoothness),
            'exterior_scale':    exterior_scale,
            'exterior_offset':   exterior_offset,
            'mask_stdamp':       mask_stdamp,
            'mask_max_distance': mask_max_distance,
        }

        candidates = [c for c in input_data['cover'].solution if c.fg_fragment.any()]
        params_id  = ray.put(params)
        futures    = [_process_candidate.remote(cidx, c, params_id) for cidx, c in enumerate(candidates)]

        postprocessed_candidates = []
        rejection_causes = {}
        for ret_idx, ret in enumerate(gocell.aux.get_ray_1by1(futures)):
            candidate, candidate_results = candidates[ret[0]], ret[1]
            candidate = PostprocessedCandidate(candidate)

            if candidate_results['fg_fragment'] is not None and candidate_results['fg_offset'] is not None:
                candidate.fg_fragment = candidate_results['fg_fragment'].copy()
                candidate.fg_offset   = candidate_results['fg_offset'  ].copy()
                if not candidate.fg_fragment.any():
                    rejection_causes[candidate] = f'empty foreground'
                    continue

            obj_radius = math.sqrt(candidate.fg_fragment.sum() / math.pi)

            if candidate_results['energy_rate'] > max_energy_rate:
                rejection_causes[candidate] = f'energy rate too high ({candidate_results["energy_rate"]})'
                continue
            if candidate_results['contrast_response'] < min_contrast_response:
                rejection_causes[candidate] = f'contrast response too low ({candidate_results["contrast_response"]})'
                continue
            if candidate.original.on_boundary:
                if discard_image_boundary or not(min_boundary_obj_radius <= obj_radius < max_obj_radius):
                    rejection_causes[candidate] = f'boundary object and/or too small/large (radius: {obj_radius})'
                    continue
            else:
                if not min_obj_radius <= obj_radius <= max_obj_radius:
                    rejection_causes[candidate] = f'object too small/large (radius: {obj_radius})'
                    continue

            postprocessed_candidates.append(candidate)
            out.intermediate(f'Post-processing candidates... {ret_idx + 1} / {len(futures)}')

        log_filename = gocell.aux.join_path(log_root_dir, 'postprocessing.txt')
        with open(log_filename, 'w') as log_file:
            for c, cause in rejection_causes.items():
                location = (c.fg_offset + np.divide(c.fg_fragment.shape, 2)).round().astype(int)
                log_line = f'object at x={location[1]}, y={location[0]} discarded: {cause}'
                log_file.write(f'{log_line}{os.linesep}')

        out.write(f'Remaining candidates: {len(postprocessed_candidates)} of {len(candidates)}')

        return {
            'postprocessed_candidates': postprocessed_candidates
        }


class PostprocessedCandidate(gocell.candidates.BaseCandidate):
    def __init__(self, original):
        self.original    = original
        self.fg_offset   = original.fg_offset
        self.fg_fragment = original.fg_fragment


def _compute_contrast_response(candidate, g, exterior_scale, exterior_offset):
    g = g / g.std()
    mask = np.zeros(g.shape, bool)
    candidate.fill_foreground(mask)
    interior_mean = g[mask].mean()
    exterior_distance_map = (ndi.distance_transform_edt(~mask) - exterior_offset).clip(0, np.inf) / exterior_scale
    exterior_mask = np.logical_xor(mask, exterior_distance_map <= 5)
    exterior_mask = np.logical_and(exterior_mask, g < interior_mean)
    exterior_weights = np.zeros(g.shape)
    exterior_weights[exterior_mask] = np.exp(-exterior_distance_map[exterior_mask])
    exterior_weights /= exterior_weights.sum()
    exterior_mean = (g * exterior_weights).sum()
    return interior_mean / exterior_mean - 1


@ray.remote
def _process_candidate(cidx, candidate, params):
    region      = candidate.get_modelfit_region(params['y'], params['g_atoms'])
    energy_rate = candidate.energy / region.mask.sum()
    contrast_response = _compute_contrast_response(candidate, params['g'], params['exterior_scale'])
    fg_offset, fg_fragment = _process_mask(candidate, params['g_smooth'], params['mask_max_distance'], params['mask_stdamp'])
    return cidx, {
        'energy_rate':       energy_rate,
        'contrast_response': contrast_response,
        'fg_offset':         fg_offset,
        'fg_fragment':       fg_fragment
    }


def _retain_intersections(superset_mask, subset_mask, copy=False):
    result = superset_mask.copy() if copy else superset_mask
    supersets = ndi.label(superset_mask)[0]
    for l in frozenset(supersets.reshape(-1)) - {0}:
        cc = (supersets == l)
        if not subset_mask[cc].any(): result[cc] = False
    return result


def _process_mask(candidate, g, max_distance, stdamp):
    if stdamp <= 0 or max_distance <= 0:
        return None, None
    mask = np.zeros(g.shape, bool)
    candidate.fill_foreground(mask)
    boundary_mask = np.logical_xor(morph.binary_dilation(mask, morph.disk(1)), morph.binary_erosion(mask, morph.disk(1)))
    boundary_distance   = 1 + ndi.distance_transform_edt(~boundary_mask)
    extra_mask_superset = (boundary_distance <= max_distance)
    g_fg_data = g[mask]
    fg_mean   = g_fg_data.mean()
    fg_amp    = g_fg_data.std() * stdamp
    extra_fg  = np.logical_and(fg_mean - fg_amp <= g, g <= fg_mean + fg_amp)
    extra_bg  = np.logical_not(extra_fg)
    extra_fg  = np.logical_and(extra_mask_superset, extra_fg)
    extra_bg  = np.logical_and(extra_mask_superset, extra_bg)
    extra_fg  = _retain_intersections(extra_fg, boundary_mask)
    extra_bg  = _retain_intersections(extra_bg, boundary_mask)
    mask[extra_fg] = True
    mask[extra_bg] = False
    fg_offset, fg_fragment = gocell.candidates.extract_foreground_fragment(mask)
    return fg_offset, fg_fragment


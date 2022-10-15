from .pipeline import Stage
from .config import get_config_value
from .candidates import BaseCandidate, extract_foreground_fragment
from ._aux import get_ray_1by1, join_path

import scipy.ndimage      as ndi
import skimage.morphology as morph
import skimage.measure

import ray
import math, os
import numpy as np


class Postprocessing(Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['cover', 'y_surface', 'g_atoms', 'g_raw'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        # simple post-processing
        max_energy_rate           = get_config_value(cfg,           'max_energy_rate',  np.inf)
        discard_image_boundary    = get_config_value(cfg,    'discard_image_boundary',   False)
        min_boundary_obj_radius   = get_config_value(cfg,   'min_boundary_obj_radius',       0)
        min_obj_radius            = get_config_value(cfg,            'min_obj_radius',       0)
        max_obj_radius            = get_config_value(cfg,            'max_obj_radius',  np.inf)
        max_eccentricity          = get_config_value(cfg,          'max_eccentricity',       1)
        max_boundary_eccentricity = get_config_value(cfg, 'max_boundary_eccentricity',  np.inf)
        if max_boundary_eccentricity is None: max_boundary_eccentricity = max_eccentricity

        # contrast-based post-processing
        get_default_contrast_response_epsilon = lambda version: {1: 0, 2: 1e-4}[version]
        contrast_response_version = get_config_value(cfg, 'contrast_response_version',  2)
        exterior_scale            = get_config_value(cfg,            'exterior_scale',  5)
        exterior_offset           = get_config_value(cfg,           'exterior_offset',  5)
        min_contrast_response     = get_config_value(cfg,     'min_contrast_response', -np.inf)
        contrast_response_epsilon = get_config_value(cfg, 'contrast_response_epsilon',  get_default_contrast_response_epsilon(contrast_response_version))

        # mask-based post-processing
        mask_stdamp          = get_config_value(cfg,          'mask_stdamp',    2)
        mask_max_distance    = get_config_value(cfg,    'mask_max_distance',    0)
        mask_smoothness      = get_config_value(cfg,      'mask_smoothness',    3)
        fill_holes           = get_config_value(cfg,           'fill_holes', True)
        mask_process_version = get_config_value(cfg, 'mask_process_version',    1)

        # autofluorescence glare removal
        glare_detection_smoothness = get_config_value(cfg, 'glare_detection_smoothness',      3)
        glare_detection_num_layers = get_config_value(cfg, 'glare_detection_num_layers',      5)
        glare_detection_min_layer  = get_config_value(cfg,  'glare_detection_min_layer',    0.5)
        min_glare_radius           = get_config_value(cfg,           'min_glare_radius', np.inf)
        min_boundary_glare_radius  = get_config_value(cfg,  'min_boundary_glare_radius', min_glare_radius)

        # mask image pixels allowed for estimation of mean background intesity during contrast computation
        background_mask = np.zeros(input_data['g_raw'].shape, bool)
        for c in input_data['cover'].solution:
            c.fill_foreground(background_mask)
        background_mask = morph.binary_erosion(~background_mask, morph.disk(exterior_offset))

        params = {
            'y':                          input_data['y_surface'],
            'g':                          input_data['g_raw'],
            'g_atoms':                    input_data['g_atoms'],
            'g_mask_processing':          ndi.gaussian_filter(input_data['g_raw'], mask_smoothness),
            'g_glare_detection':          ndi.gaussian_filter(input_data['g_raw'], glare_detection_smoothness),
            'contrast_response_version':  contrast_response_version,
            'background_mask':            background_mask,
            'exterior_scale':             exterior_scale,
            'exterior_offset':            exterior_offset,
            'contrast_response_epsilon':  contrast_response_epsilon,
            'mask_stdamp':                mask_stdamp,
            'mask_max_distance':          mask_max_distance,
            'mask_process_version':       mask_process_version,
            'fill_holes':                 fill_holes,
            'min_glare_radius':           min_glare_radius,
            'min_boundary_glare_radius':  min_boundary_glare_radius,
            'glare_detection_min_layer':  glare_detection_min_layer,
            'glare_detection_num_layers': glare_detection_num_layers
        }

        candidates = [c for c in input_data['cover'].solution if c.fg_fragment.any()]
        params_id  = ray.put(params)
        futures    = [_process_candidate.remote(cidx, c, params_id) for cidx, c in enumerate(candidates)]

        postprocessed_candidates = []
        log_entries = []
        for ret_idx, ret in enumerate(get_ray_1by1(futures)):
            candidate, candidate_results = candidates[ret[0]], ret[1]
            candidate = PostprocessedCandidate(candidate)

            if candidate_results['fg_fragment'] is not None and candidate_results['fg_offset'] is not None:
                candidate.fg_fragment = candidate_results['fg_fragment'].copy()
                candidate.fg_offset   = candidate_results['fg_offset'  ].copy()
                if not candidate.fg_fragment.any():
                    log_entries.append((candidate, f'empty foreground'))
                    continue

            if candidate_results['is_glare']:
                log_entries.append((candidate, f'glare removed (radius: {candidate_results["obj_radius"]})'))
                continue
            if candidate_results['energy_rate'] > max_energy_rate:
                log_entries.append((candidate, f'energy rate too high ({candidate_results["energy_rate"]})'))
                continue
            if candidate_results['contrast_response'] < min_contrast_response:
                log_entries.append((candidate, f'contrast response too low ({candidate_results["contrast_response"]})'))
                continue
            if candidate.original.on_boundary:
                if candidate_results['eccentricity'] > max_boundary_eccentricity:
                    log_entries.append((candidate, f'boundary object eccentricity too high ({candidate_results["eccentricity"]})'))
                    continue
                if discard_image_boundary:
                    log_entries.append((candidate, f'boundary object discarded'))
                    continue
                if not(min_boundary_obj_radius <= candidate_results['obj_radius'] <= max_obj_radius):
                    log_entries.append((candidate, f'boundary object and/or too small/large (radius: {candidate_results["obj_radius"]})'))
                    continue
            else:
                if candidate_results['eccentricity'] > max_eccentricity:
                    log_entries.append((candidate, f'eccentricity too high ({candidate_results["eccentricity"]})'))
                    continue
                if not min_obj_radius <= candidate_results['obj_radius'] <= max_obj_radius:
                    log_entries.append((candidate, f'object too small/large (radius: {candidate_results["obj_radius"]})'))
                    continue

            postprocessed_candidates.append(candidate)
            out.intermediate(f'Post-processing candidates... {ret_idx + 1} / {len(futures)}')

        if log_root_dir is not None:
            log_filename = join_path(log_root_dir, 'postprocessing.txt')
            with open(log_filename, 'w') as log_file:
                for c, comment in log_entries:
                    location = (c.fg_offset + np.divide(c.fg_fragment.shape, 2)).round().astype(int)
                    log_line = f'object at x={location[1]}, y={location[0]}: {comment}'
                    log_file.write(f'{log_line}{os.linesep}')

        out.write(f'Remaining candidates: {len(postprocessed_candidates)} of {len(candidates)}')

        return {
            'postprocessed_candidates': postprocessed_candidates
        }


class PostprocessedCandidate(BaseCandidate):
    def __init__(self, original):
        self.original    = original
        self.fg_offset   = original.fg_offset
        self.fg_fragment = original.fg_fragment


def _compute_contrast_response(candidate, g, exterior_scale, exterior_offset, epsilon, background_mask, version):
    assert version in [1,2]
    g = g / g.std()
    mask = np.zeros(g.shape, bool)
    candidate.fill_foreground(mask)
    interior_mean = g[mask].mean()
    exterior_distance_map = (ndi.distance_transform_edt(~mask) - exterior_offset).clip(0, np.inf) / exterior_scale
    exterior_mask = np.logical_xor(mask, exterior_distance_map <= 5)
    if version == 1:
        exterior_mask = np.logical_and(exterior_mask, g < interior_mean)
    elif version == 2:
        exterior_mask = np.logical_and(exterior_mask, background_mask)
    exterior_weights = np.zeros(g.shape)
    exterior_weights[exterior_mask] = np.exp(-exterior_distance_map[exterior_mask])
    exterior_weights /= exterior_weights.sum()
    exterior_mean = (g * exterior_weights).sum()
    if version == 1:
        return interior_mean / (exterior_mean + epsilon) - 1
    elif version == 2:
        return (interior_mean + epsilon) / (exterior_mean + epsilon) - 1


def _is_glare(candidate, g, min_layer=0.5, num_layers=5):
    g_sect = g[candidate.fg_offset[0] : candidate.fg_offset[0] + candidate.fg_fragment.shape[0],
               candidate.fg_offset[1] : candidate.fg_offset[1] + candidate.fg_fragment.shape[1]]
    mask = morph.binary_erosion(candidate.fg_fragment, morph.disk(2))
    g_sect_data = g_sect[mask]
    get_layer   = lambda prop: np.logical_and(mask, g_sect > (g_sect_data.max() - g_sect_data.min()) * prop + g_sect_data.min())
    count_cc    = lambda mask: ndi.label(mask)[0].max()
    is_subset   = lambda mask_sub, mask_sup: (np.logical_or(mask_sub, mask_sup) == mask_sub).all()
    props       = np.linspace(min_layer, 1, num_layers, endpoint=False)
    prev_layer  = None
    is_glare    = True
    for prop in props:
        layer = get_layer(prop)
        if count_cc(layer) > 1:
            is_glare = False
            break
        prev_layer = layer
    return is_glare


def _compute_energy_rate(candidate, y, g_atoms):
    region = candidate.get_modelfit_region(y, g_atoms)
    return candidate.energy / region.mask.sum()


@ray.remote
def _process_candidate(cidx, candidate, params):
    obj_radius = math.sqrt(candidate.fg_fragment.sum() / math.pi)
    is_glare   = False
    if params['min_boundary_glare_radius' if candidate.on_boundary else 'min_glare_radius'] < obj_radius:
        is_glare = _is_glare(candidate, params['g_glare_detection'], params['glare_detection_min_layer'], params['glare_detection_num_layers'])
    energy_rate  = _compute_energy_rate(candidate, params['y'], params['g_atoms'])
    contrast_response = _compute_contrast_response(candidate, params['g'], params['exterior_scale'], params['exterior_offset'], params['contrast_response_epsilon'], params['background_mask'], params['contrast_response_version'])
    fg_offset, fg_fragment = _process_mask(candidate, params['g_mask_processing'], params['mask_max_distance'], params['mask_stdamp'], params['fill_holes'], params['mask_process_version'])
    eccentricity = _compute_eccentricity(candidate)

    return cidx, {
        'energy_rate':       energy_rate,
        'contrast_response': contrast_response,
        'fg_offset':         fg_offset,
        'fg_fragment':       fg_fragment,
        'obj_radius':        obj_radius,
        'is_glare':          is_glare,
        'eccentricity':      eccentricity
    }


def _process_mask(candidate, g, max_distance, stdamp, fill_holes=False, version=1):
    assert version in (1,2)
    if stdamp <= 0 or max_distance <= 0:
        if fill_holes:
            return candidate.fg_offset, ndi.morphology.binary_fill_holes(candidate.fg_fragment)
        else:
            return None, None
    mask = np.zeros(g.shape, bool)
    candidate.fill_foreground(mask)
    extra_mask_superset = np.logical_xor(morph.binary_dilation(mask, morph.disk(max_distance)), morph.binary_erosion(mask, morph.disk(max_distance)))
    g_fg_data = g[mask]
    fg_mean   = g_fg_data.mean()
    fg_amp    = g_fg_data.std() * stdamp
    extra_fg  = np.logical_and(fg_mean - fg_amp <= g, g <= fg_mean + fg_amp)
    extra_bg  = np.logical_not(extra_fg)
    extra_fg  = np.logical_and(extra_mask_superset, extra_fg)
    extra_bg  = np.logical_and(extra_mask_superset, extra_bg)
    
    if version >= 2:
        extra_fg_labels = ndi.label(extra_fg)[0]
        for label in np.unique(extra_fg_labels):
            if label == 0: continue
            extra_fg_cc = (extra_fg_labels == label)
            if not mask[extra_fg_cc].any(): extra_fg[extra_fg_cc] = False
        
    mask[extra_fg] = True
    mask[extra_bg] = False
    fg_offset, fg_fragment = extract_foreground_fragment(mask)
    if fill_holes: fg_fragment = ndi.morphology.binary_fill_holes(fg_fragment)
    return fg_offset, fg_fragment


def _compute_eccentricity(candidate):
    if candidate.fg_fragment.any():
        return skimage.measure.regionprops(candidate.fg_fragment.astype('uint8'), coordinates='rc')[0].eccentricity
    else:
        return 0


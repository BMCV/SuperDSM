from .pipeline import Stage
from .candidates import BaseCandidate, extract_foreground_fragment
from ._aux import get_ray_1by1, join_path

import scipy.ndimage      as ndi
import skimage.morphology as morph
import skimage.measure

import ray
import math, os
import numpy as np


class Postprocessing(Stage):
    """Implements the post-processing as described in Section 3.4 and Supplemental Material 7 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`).

    Hyperparameters
    ---------------

    The following hyperparameters can be used to control this pipeline stage.

    Simple post-processing
    ^^^^^^^^^^^^^^^^^^^^^^

    ``postprocess/max_energy_rate``
        Objects with a normalized energy larger than this value are discarded. Corresponds to ``max_norm_energy2`` in the paper (Supplemental Material 8, also incorrectly referred to as ``min_norm_energy2`` in Supplemental Material 7 due to a typo).

    ``postprocess/discard_image_boundary``
        Objects located directly on the image border are discarded if this is set to ``True``.

    ``postprocess/min_object_radius``
        Objects smaller than a circle of this radius are discarded. Defaults to 0.

    ``postprocess/max_object_radius``
        Objects larger than a circle of this radius are discarded. Defaults to infinity.

    ``postprocess/min_boundary_obj_radius``
        Overrides ``postprocess/min_object_radius`` for objects located directly on the image border. Defaults to the value of the ``postprocess/min_object_radius`` hyperparameter.

    ``postprocess/max_eccentricity``
        Objects with an eccentricity higher than this value are discarded. Defaults to 0.99.

    ``postprocess/max_boundary_eccentricity``
        Overrides ``postprocess/max_boundary_eccentricity`` for objects located directly on the image border. Defaults to the value set for ``postprocess/max_boundary_eccentricity``.

    Contrast-based post-processing
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A segmented object is discarded, if the *contrast response* is too low, which is the ratio of *(i)* the mean image intensity inside the segmentation mask of the object and *(ii)* the average image intensity within its exterior neighborhood. The average image intensity within the exterior neighborhood is determined using a weighted mean of the image intensities, where the weight

    .. math:: \\exp(-\\max\\{0, \\operatorname{dist}_M(x) - \\text{exterior_offset}\\} / \\text{exterior_scale})
        
    of an image point :math:`x` decays exponentially with the Euclidean distance :math:`\\operatorname{dist}_M(x)` of that point to the mask :math:`M` of the segmented object. Image points corresponding to segmentation masks of segmented objects are weighted by zero.

    ``postprocess/exterior_scale``
        Scales the thickness of the soft margin of the exterior neighborhood (this is the *outer* margin, corresponding to image points associated with weights smaller than 1). Increasing this value increases the importance of image points further away of the segmentation mask. Defaults to 5.

    ``postprocess/exterior_offset``
        Corresponds to the thickness of the *inner* margin of image points within the exterior neighborhood which are weighted by 1. Increasing this value increases the importance of image points closest to the segmentation mask. Defaults to 5.

    ``postprocess/min_contrast_response``
        A segmented object is discarded, if the contrast response as defined above is below this threshold. Corresponds to ``min_contrast`` in the paper (Supplemental Material 7 and 8). Defaults to 1.35.

    ``postprocess/contrast_response_epsilon``
        This constant is added to both the nominator and the denominator of the fraction which defines the contrast response (see above). Defaults to 1e-4.

    Mask-based post-processing
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    The segmentation masks are refined individually (independently of each other).

    ``postprocess/mask_stdamp``
        An image point adjacent to the boundary of the original segmentation mask is added to the segmentation mask, if its Gaussian-smoothed intensity is sufficiently similar to the mean intensity of the mask. The image point is removed otherwise. The lower the value set for ``postprocess/mask_stdamp``, the stricter the similarity must be. Defaults to 2.

    ``postprocess/mask_max_distance``
        Image points within this maximum distance of the boundary of the original segmentation mask are subject to refinement. Image points further away from the boundary are neither added to nor removed from the segmentation mask. Defaults to 1.

    ``postprocess/mask_smoothness``
        Corresponds to the scale of the Gaussian filter used to smooth the image intensities for refinement of the segmentation mask. Defaults to 3.

    ``postprocess/fill_holes``
        Morphological holes in the segmentation mask are filled if set to ``True``. Defaults to ``True``.

    Autofluorescence glare removal
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    To decide whether a segmented object is an autofluorescence artifact, we considere its segmentation mask. The object is identified as an autofluorescence artifiact and discarded, if it is sufficiently large and, by default, the top 50% of the Gaussian-smoothed intensity profile of the object mask is approximately pyramid-shaped (i.e. the top 50% intensity-superlevel sets of the object are connected). This is illustrated in the figure below, where the intensity-superlevel sets are shown for 5 different intensity values:

    .. figure:: glare_detection.png
       :width: 100%

       Autofluorescence artifact detection method. (a) Original image section (NIH3T3 cells, contrast-enhanced). (b) Ground truth segmentation. (c) Segmentation result of cell nuclei (green contour) and the autofluorescence artifact (red contour). (d) Smoothed image intensities (Gaussian filter) and the corresponding intensity profiles (solid contours) of the detected objects (dashed contours).

    ``postprocess/glare_detection_smoothness``
        The standard deviation of the Gaussian function used for smoothing the image intensities of the segmented object. Defaults to 3.

    ``postprocess/glare_detection_num_layers``
        The number of intensity values, for which the connectivity of the intensity-superlevel sets is investigated. Defaults to 5.

    ``postprocess/glare_detection_min_layer``
        The top fraction of the Gaussian-smoothed intensity profile investigated for connectivity. Defaults to 0.5, i.e. the top 50% of the Gaussian-smoothed intensity profile is investigated.

    ``postprocess/min_glare_radius``
        The size of a segmented object must correspond to a circle at least of this radius in order for the object to be possibly recognized as an autofluorescence artifact. Defaults to infinity.

    ``postprocess/min_boundary_glare_radius``
        Overrides ``postprocess/min_glare_radius`` for objects located directly on the image border. Defaults to the value of the ``postprocess/min_glare_radius`` hyperparameter.
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(Postprocessing, self).__init__('postprocess',
                                             inputs  = ['cover', 'y_surface', 'g_atoms', 'g_raw'],
                                             outputs = ['postprocessed_candidates'])

    def process(self, input_data, cfg, out, log_root_dir):
        # simple post-processing
        max_energy_rate           = cfg.get(          'max_energy_rate',    0.2)
        discard_image_boundary    = cfg.get(   'discard_image_boundary',  False)
        min_boundary_obj_radius   = cfg.get(  'min_boundary_obj_radius',      0)
        min_obj_radius            = cfg.get(        'min_object_radius',      0)
        max_obj_radius            = cfg.get(        'max_object_radius', np.inf)
        max_eccentricity          = cfg.get(         'max_eccentricity',   0.99)
        max_boundary_eccentricity = cfg.get('max_boundary_eccentricity', np.inf)
        if max_boundary_eccentricity is None: max_boundary_eccentricity = max_eccentricity

        # contrast-based post-processing
        exterior_scale            = cfg.get(           'exterior_scale',    5)
        exterior_offset           = cfg.get(          'exterior_offset',    5)
        min_contrast_response     = cfg.get(    'min_contrast_response', 1.35)
        contrast_response_epsilon = cfg.get('contrast_response_epsilon', 1e-4)

        # mask-based post-processing
        mask_stdamp          = cfg.get(      'mask_stdamp',    2)
        mask_max_distance    = cfg.get('mask_max_distance',    1)
        mask_smoothness      = cfg.get(  'mask_smoothness',    3)
        fill_holes           = cfg.get(       'fill_holes', True)

        # autofluorescence glare removal
        glare_detection_smoothness = cfg.get('glare_detection_smoothness',      3)
        glare_detection_num_layers = cfg.get('glare_detection_num_layers',      5)
        glare_detection_min_layer  = cfg.get( 'glare_detection_min_layer',    0.5)
        min_glare_radius           = cfg.get(          'min_glare_radius', np.inf)
        min_boundary_glare_radius  = cfg.get( 'min_boundary_glare_radius', min_glare_radius)

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
            'background_mask':            background_mask,
            'exterior_scale':             exterior_scale,
            'exterior_offset':            exterior_offset,
            'contrast_response_epsilon':  contrast_response_epsilon,
            'mask_stdamp':                mask_stdamp,
            'mask_max_distance':          mask_max_distance,
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


def _compute_contrast_response(candidate, g, exterior_scale, exterior_offset, epsilon, background_mask):
    g = g / g.std()
    mask = np.zeros(g.shape, bool)
    candidate.fill_foreground(mask)
    interior_mean = g[mask].mean()
    exterior_distance_map = (ndi.distance_transform_edt(~mask) - exterior_offset).clip(0, np.inf) / exterior_scale
    exterior_mask = np.logical_xor(mask, exterior_distance_map <= 5)
    exterior_mask = np.logical_and(exterior_mask, background_mask)
    exterior_weights = np.zeros(g.shape)
    exterior_weights[exterior_mask] = np.exp(-exterior_distance_map[exterior_mask])
    exterior_weights /= exterior_weights.sum()
    exterior_mean = (g * exterior_weights).sum()
    return (interior_mean + epsilon) / (exterior_mean + epsilon)


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
    contrast_response = _compute_contrast_response(candidate, params['g'], params['exterior_scale'], params['exterior_offset'], params['contrast_response_epsilon'], params['background_mask'])
    fg_offset, fg_fragment = _process_mask(candidate, params['g_mask_processing'], params['mask_max_distance'], params['mask_stdamp'], params['fill_holes'])
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


def _process_mask(candidate, g, max_distance, stdamp, fill_holes=False):
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


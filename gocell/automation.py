import gocell.render
import gocell.config

import skimage
import math
import scipy.ndimage as ndi
import numpy as np


def _blob_doh(image, sigma_list, threshold=0.01, overlap=.5, mask=None):
    """Finds blobs in the given grayscale image.

    This implementation is widely based on:
    https://github.com/scikit-image/scikit-image/blob/fca9f16da4bd7420245d05fa82ee51bb9677b039/skimage/feature/blob.py#L538-L646
    """
    skimage.feature.blob.check_nD(image, 2)
    if mask is None: mask = np.ones(image.shape, bool)
    if not isinstance(mask, dict): mask = {sigma: mask for sigma in sigma_list}

    image = skimage.feature.blob.img_as_float(image)
    image = skimage.feature.blob.integral_image(image)

    hessian_images = [mask[s] * skimage.feature.blob._hessian_matrix_det(image, s) for s in sigma_list]
    image_cube = np.dstack(hessian_images)

    local_maxima = skimage.feature.blob.peak_local_max(image_cube, threshold_abs=threshold,
                                                       footprint=np.ones((3,) * image_cube.ndim),
                                                       threshold_rel=0.0,
                                                       exclude_border=False)

    if local_maxima.size == 0:
        return np.empty((0, 3))
    lm = local_maxima.astype(np.float64)
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return skimage.feature.blob._prune_blobs(lm, overlap)


def _estimate_scale(im, min_radius=20, max_radius=200, num_radii=10, thresholds=[0.01], inlier_tol=np.inf):
    """Estimates the scale of the image.
    """

    sigma_list = np.linspace(min_radius, max_radius, num_radii) / math.sqrt(2)
    sigma_list = np.concatenate([[sigma_list.min() / 2], sigma_list])
    
    im_norm  = gocell.render.normalize_image(im)
    im_norm /= im_norm.max()

    blobs_mask  = {sigma: ndi.gaussian_laplace(im_norm, sigma) < 0 for sigma in sigma_list}
    mean_radius = None
    for threshold in sorted(thresholds, reverse=True):
        blobs_doh = _blob_doh(im_norm, sigma_list, threshold=threshold, mask=blobs_mask)
        blobs_doh = blobs_doh[~np.isclose(blobs_doh[:,2], sigma_list.min())]
        if len(blobs_doh) == 0: continue

        radii = blobs_doh[:,2] * math.sqrt(2)
        radii_median  = np.median(radii)
        radii_mad     = np.mean(np.abs(radii - np.median(radii)))
        radii_bound   = np.inf if np.isinf(inlier_tol) else redii_mad * inlier_tol
        radii_inliers = np.logical_and(radii >= radii_median - radii_mad, radii <= radii_median + radii_mad)
        mean_radius   = np.mean(radii[radii_inliers])
        break
    
    if mean_radius is None:
        raise ValueError('scale estimation failed')
    return mean_radius / math.sqrt(2), blobs_doh, radii_inliers


def _create_config_entry(version, cfg, key, factor, default_user_factor, type=None, _min=None, _max=None):
    keys = key.split('/')
    af_key = f'{"/".join(keys[:-1])}/AF_{keys[-1]}'
    override_none = (version >= 3)
    gocell.config.set_default_value(cfg, key, factor * gocell.config.get_value(cfg, af_key, default_user_factor), override_none)
    if type is not None: gocell.config.update_value(cfg, key, func=type)
    if _min is not None: gocell.config.update_value(cfg, key, func=lambda value: max((value, _min)))
    if _max is not None: gocell.config.update_value(cfg, key, func=lambda value: min((value, _max)))


def create_config(base_cfg, im):
    config   = gocell.config.copy(base_cfg)
    version  = gocell.config.get_value(config, 'af_version', 1)

    if version == 0:
        scale = None

    if version >= 1:
        scale = gocell.config.get_value(config, 'AF_scale', None)
        if scale is None: scale = _estimate_scale(im, num_radii=10, thresholds=[0.01])[0]
        radius   = scale * math.sqrt(2)
        diameter = 2 * radius

        _create_config_entry(version, config, 'preprocess/sigma2'            , scale      ,    1.0)
        _create_config_entry(version, config, 'generations/alpha'            , radius ** 2,    0.2)
        _create_config_entry(version, config, 'generations/max_seed_distance', diameter   , np.inf)
        _create_config_entry(version, config, 'postprocess/min_obj_radius'   , radius     ,    0.0)
        _create_config_entry(version, config, 'postprocess/max_obj_radius'   , radius     , np.inf)
        _create_config_entry(version, config, 'postprocess/min_glare_radius' , radius     , np.inf)
    
    if version >= 2:
        _create_config_entry(version, config, 'modelfit/rho'             , scale ** 2, 0.015 ** 2)
        _create_config_entry(version, config, 'modelfit/smooth_amount'   , scale     , 0.2, type=int, _min=4)
        _create_config_entry(version, config, 'modelfit/smooth_subsample', scale     , 0.4, type=int, _min=8)
    
    if version >= 3:
        _create_config_entry(version, config, 'top-down-segmentation/min_region_radius', radius, 0.25, type=int)
    
    if version >= 4:
        _create_config_entry(version, config, 'top-down-modelfit/min_background_margin', scale, 0.2, type=int)

    return config, scale


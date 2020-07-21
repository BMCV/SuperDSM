import gocell.aux

import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


class Candidate:
    def __init__(self):
        self.fg_offset   = None
        self.fg_fragment = None
        self.footprint   = set()
        self.energy      = np.nan
        self.on_boundary = np.nan
    
    def get_mask(self, g_atoms):
        return np.in1d(g_atoms, list(self.footprint)).reshape(g_atoms.shape)

    def get_region(self, g, g_superpixels):
        region_mask = self.get_mask(g_superpixels)
        return surface.Surface(g.model.shape, g.model, mask=region_mask)

    def set(self, state):
        self.fg_fragment = state.fg_fragment.copy() if state.fg_fragment is not None else None
        self.fg_offset   = state.fg_offset.copy() if state.fg_offset is not None else None
        self.footprint   = set(state.footprint)
        self.energy      = state.energy
        self.on_boundary = state.on_boundary
        return self

    def copy(self):
        return Candidate().set(self)
    
    def fill_foreground(self, out, value=True):
        sel = np.s_[self.fg_offset[0] : self.fg_offset[0] + self.fg_fragment.shape[0], self.fg_offset[1] : self.fg_offset[1] + self.fg_fragment.shape[1]]
        out[sel] = value * self.fg_fragment
        return sel


def _expand_mask_for_backbround(y, mask, margin_step, max_margin):
    img_boundary_mask = zeros(mask.shape, bool)
    img_boundary_mask[ 0,  :] = True
    img_boundary_mask[-1,  :] = True
    img_boundary_mask[ :,  0] = True
    img_boundary_mask[ :, -1] = True
    img_boundary_mask = np.logical_and(img_boundary_mask, y < 0)
    mask_foreground = np.logical_and(y > 0, mask)
    mask_boundary   = np.logical_xor(mask, morph.binary_erosion(mask, morph.disk(1)))
    if not np.logical_and(mask_foreground, mask_boundary).any():
        return np.logical_or(img_boundary_mask, mask)
    tmp11 = ndi.distance_transform_edt(~mask_foreground)
    tmp12 = tmp11 * (y < 0)
    for thres in range(margin_step, max_margin + 1, margin_step):
        expansion_mask = np.logical_and(tmp12 > 0, tmp12 <= thres)
        exterior_mask  = np.logical_and(thres < tmp11, tmp11 < thres + 1)
        tmp15 = ndi.label(~expansion_mask if thres > 0 else ~mask_boundary)[0]
        if len(frozenset(tmp15[mask_foreground]) & frozenset(tmp15[exterior_mask])) == 0: break
    tmp16 = np.logical_or(mask, np.logical_and(np.logical_or(mask, tmp11 <= thres), y < 0))
    if mask_foreground[img_boundary_mask].any():
        return np.logical_or(img_boundary_mask, tmp16)
    else: return tmp16


def _get_modelfit_region(candidate, y, g_atoms):
    region = candidate.get_region(y, g_atoms)
    bg_mask = _expand_mask_for_backbround(y.model, region.mask, 20, 500)
    region.mask = np.logical_or(region.mask, np.logical_and(y.model < 0, bg_mask))
    return region


def _process_candidate(y, g_atoms, x_map, candidate, modelfit_kwargs, silent=True):
    region = _get_modelfit_region(candidate, y, g_atoms)
    def _run(): return gocell.modelfit.modelfit(y, region, **modelfit_kwargs)
    if silent:
        with contextlib.redirect_stdout(None): J, result, fallback = _run()
    else:
        J, result, fallback = _run()
    padded_mask = np.pad(region.mask, 1)
    smooth_mat  = gocell.aux.uplift_smooth_matrix(J.smooth_mat, padded_mask)
    padded_foreground = (result.map_to_image_pixels(g, region, pad=1).s(x_map, smooth_mat) > 0)
    foreground = padded_foreground[1:-1, 1:-1]
    if foreground.any():
        rows = foreground.any(axis=1)
        cols = foreground.any(axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        candidate.fg_offset   = np.array([rmin, cmin])
        candidate.fg_fragment = foreground[rmin : rmax + 1, cmin : cmax + 1]
    else:
        candidate.fg_offset   = np.zeros(2, int)
        candidate.fg_fragment = np.zeros((1, 1), bool)
    candidate.energy      = J(result)
    candidate.on_boundary = padded_foreground[0].any() or padded_foreground[-1].any() or padded_foreground[:, 0].any() or padded_foreground[:, -1].any()
    return candidate, fallback


@ray.remote
def process_candidate(cidx, *args, **kwargs):
    return (cidx, *_process_candidate(*args, **kwargs))


def process_candidates(candidates, y, g_atoms, modelfit_kwargs, out=None):
    out = gocell.aux.get_output(out)
    candidates   = list(candidates)
    y_id         = ray.put(y)
    g_atoms_id   = ray.put(g_atoms)
    x_map_id     = ray.put(y.get_map(normalized=False, pad=1))
    mf_kwargs_id = ray.put(modelfit_kwargs)
    futures      = [process_candidate.remote(cidx, y_id, g_atoms_id, x_map_id, c, mf_kwargs_id) for cidx, c in enumerate(candidates)]
    fallbacks    = 0
    for ret_idx, ret in enumerate(gocell.aux.get_ray_1by1(futures)):
        candidates[ret[0]].set(ret[1])
        out.intermediate(f'Processing candidates... {ret_idx + 1} / {len(futures)} ({fallbacks}x fallback)')
        if ret[2]: fallbacks += 1
    out.write(f'Processed candidates: {len(candidates)} ({fallbacks}x fallback)')


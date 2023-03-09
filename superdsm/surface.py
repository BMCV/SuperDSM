from .io import imread

import numpy as np


def bbox(mask, include_end=True):
    mask_a0 = mask.any(axis=0)
    mask_a1 = mask.any(axis=1)
    ret = np.array([np.where(mask_a1)[0][[0, -1]], np.where(mask_a0)[0][[0, -1]]])
    if include_end: ret += np.array([0, 1])
    return ret, np.s_[ret[0][0] : ret[0][1], ret[1][0] : ret[1][1]]


class Surface:

    def __init__(self, model=None, mask=None, full_mask=None, offset=(0,0)):
        self.model     = model
        self.mask      = mask if mask is not None else np.ones(model.shape, bool)
        self.full_mask = full_mask if full_mask is not None else self.mask
        self.offset    = offset

    def get_region(self, mask, shrink=False):
        mask = np.logical_and(self.mask, mask)
        if shrink:
            _bbox = bbox(mask)
            return Surface(self.model[_bbox[1]], mask[_bbox[1]], full_mask=mask, offset=tuple(_bbox[0][:,0]))
        else:
            return Surface(self.model, mask)
    
    @staticmethod
    def create_from_image(img, mask=None, normalize=True):
        assert mask is None or (isinstance(mask, np.ndarray) and mask.dtype == bool)
        if normalize:
            img_diff = img.max() - img.min()
            if img_diff == 0: img_diff = 1
            img = (img - img.min()).astype(float) / img_diff
        return Surface(model=img, mask=mask)

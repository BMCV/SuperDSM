from .io import imread

import numpy as np


def get_pixel_map(shape, normalized=False):
    z = (np.array(shape) - 1. if normalized else np.ones(2))[Ellipsis, None, None]
    z[z == 0] = 1
    return np.indices(shape) / z


def bbox(mask, include_end=True):
    mask_a0 = mask.any(axis=0)
    mask_a1 = mask.any(axis=1)
    ret = np.array([np.where(mask_a1)[0][[0, -1]], np.where(mask_a0)[0][[0, -1]]])
    if include_end: ret += np.array([0, 1])
    return ret, np.s_[ret[0][0] : ret[0][1], ret[1][0] : ret[1][1]]


def normalize_image(img):
    """Normalizes the image intensities to the range from 0 to 1.

    The original image ``img`` is not modified.

    :return: The normalized image.
    """
    img_diff = img.max() - img.min()
    if img_diff == 0: img_diff = 1
    return (img - img.min()).astype(float) / img_diff

class Image:
    """This class is used internally in SuperDSM to ease the work with images and image regions.
    """

    def __init__(self, model=None, mask=None, full_mask=None, offset=(0,0)):
        self.model     = model
        self.mask      = mask if mask is not None else np.ones(model.shape, bool)
        self.full_mask = full_mask if full_mask is not None else self.mask
        self.offset    = offset

    def shrink_mask(self, mask):
        return mask[self.offset[0] : self.offset[0] + self.mask.shape[0],
                    self.offset[1] : self.offset[1] + self.mask.shape[1]]

    def get_region(self, mask, shrink=False):
        mask = np.logical_and(self.mask, mask)
        if shrink:
            _bbox = bbox(mask)
            return Image(self.model[_bbox[1]], mask[_bbox[1]], full_mask=mask, offset=tuple(_bbox[0][:,0]))
        else:
            return Image(self.model, mask)
    
    @staticmethod
    def create_from_array(img, mask=None, normalize=True):
        assert mask is None or (isinstance(mask, np.ndarray) and mask.dtype == bool)
        if normalize: img = normalize_image(img)
        return Image(model=img, mask=mask)

    def get_map(self, normalized=True, pad=0):
        assert pad >= 0 and isinstance(pad, int)
        return get_pixel_map(np.add(self.model.shape, 2 * pad), normalized)

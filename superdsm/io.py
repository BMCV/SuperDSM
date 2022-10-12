import skimage.io, skimage.transform
import os, warnings
import ._warps as _warps


def imwrite(filepath, img, shape=None, antialias=False):
    if shape is not None:
        aa, aa_sigma = False, None
        img = img.astype(float)
        order = 0
        if antialias is not None:
            if isinstance(antialias, float):
                aa_sigma = antialias
                aa = True
                order = 1
            elif isinstance(antialias, bool):
                aa = antialias
                order = 1 if antialias else 0
        img = _warps.resize(img, shape, order=order, anti_aliasing=aa, anti_aliasing_sigma=aa_sigma, mode='reflect')
    filepath = os.path.expanduser(filepath)
    if str(img.dtype).startswith('float'):
        img = (img - img.min()) / (img.max() - img.min()) 
        img = (img * 255).round().astype('uint8')
    skimage.io.imsave(filepath, img)


def imread(filepath, **kwargs):
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise ValueError('not a file: %s' % filepath)
    fp_lowercase = filepath.lower()
    if 'as_gray' not in kwargs: kwargs['as_gray'] = True
    if fp_lowercase.endswith('.png'):
        img = skimage.io.imread(filepath, **kwargs)
    elif fp_lowercase.endswith('.tif') or fp_lowercase.endswith('.tiff'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            img = skimage.io.imread(filepath, plugin='tifffile', **kwargs)
    else:
        raise ValueError('unknown file extension: %s' % filepath)
    return img


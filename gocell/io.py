import skimage.io, skimage.transform
import os
import _warps


def imwrite(filepath, img, shape=None, antialias=False):
    if shape is not None:
        aa, aa_sigma = False, None
        img = img.astype(float)
        if antialias is not None:
            if isinstance(antialias, float):
                aa_sigma = antialias
                aa = True
            elif isinstance(antialias, bool):
                aa = antialias
        img = _warps.resize(img, shape, anti_aliasing=aa, anti_aliasing_sigma=aa_sigma, mode='reflect')
    filepath = os.path.expanduser(filepath)
    if str(img.dtype).startswith('float'):
        img = (img - img.min()) / (img.max() - img.min()) 
        img = (img * 255).round().astype('uint8')
    kwargs = {'format_str': 'PNG'} if filepath.lower().endswith('.png') else {}
    skimage.io.imsave(filepath, img, **kwargs)


def imread(filepath, **kwargs):
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise ValueError('not a file: %s' % filepath)
    fp_lowercase = filepath.lower()
    if fp_lowercase.endswith('.png'):
        img = skimage.io.imread(filepath, as_grey=True, **kwargs)
    elif fp_lowercase.endswith('.tif') or fp_lowercase.endswith('.tiff'):
        img = skimage.io.imread(filepath, as_grey=True, plugin='tifffile', **kwargs)
    else:
        raise ValueError('unknown file extension: %s' % filepath)
    return img


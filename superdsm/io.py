import skimage.io, skimage.transform._warps
import os, warnings


def imsave(filepath, img, shape=None, antialias=False):
    """
    Writes an image to a file.

    :param filepath: The path of the file to be written.
    :param img: A ``numpy.ndarray`` object corresponding to the image data.
    :param shape: Resolution of the image to be written. Useful for up- and downsampling the image data.
    :param antialias: Whether interpolation and/or anti aliasing should be used for resampling (only used if ``shape`` is not ``None``).
    """
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
        img = skimage.transform._warps.resize(img, shape, order=order, anti_aliasing=aa, anti_aliasing_sigma=aa_sigma, mode='reflect')
    filepath = os.path.expanduser(filepath)
    if str(img.dtype).startswith('float'):
        img = (img - img.min()) / (img.max() - img.min()) 
        img = (img * 255).round().astype('uint8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        skimage.io.imsave(filepath, img)


def imread(filepath, force_filetype=None, **kwargs):
    """
    Loads an image from file.

    Supported file extensions are PNG, TIF, and TIFF.

    :param force_filetype: Pretend that the file has a specific extension.
    """

    if force_filetype is not None:
        force_filetype = force_filetype.lower()
        assert force_filetype in ('png', 'tif', 'tiff')
        filetype = force_filetype

    else:
        filepath_parts = str(filepath).split('.')
        assert len(filepath_parts) >= 2, f'Failed to determine file extension: {filepath}'
        filetype = filepath_parts[-1].lower()

    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise ValueError(f'Not a file: {filepath}')

    if 'as_gray' not in kwargs:
        kwargs['as_gray'] = True

    if filetype == 'png':
        img = skimage.io.imread(filepath, **kwargs)

    elif filetype in ('tif', 'tiff'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            img = skimage.io.imread(filepath, plugin='tifffile', **kwargs)

    else:
        raise ValueError(f'Unknown file extension: {filepath}')
    return img

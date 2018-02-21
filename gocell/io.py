import skimage.io
import os


def imwrite(filepath, img):
    filepath = os.path.expanduser(filepath)
    if str(img.dtype).startswith('float'):
        img = (img - img.min()) / (img.max() - img.min()) 
        img = (img * 255).round().astype('uint8')
    kwargs = {'format_str': 'PNG'} if filepath.lower().endswith('.png') else {}
    skimage.io.imsave(filepath, img, **kwargs)


def imread(filepath):
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise ValueError('not a file: %s' % filepath)
    fp_lowercase = filepath.lower()
    if fp_lowercase.endswith('.png'):
        img = skimage.io.imread(filepath, as_grey=True)
    elif fp_lowercase.endswith('.tif') or fp_lowercase.endswith('.tiff'):
        img = skimage.io.imread(filepath, as_grey=True, plugin='tifffile')
    else:
        raise ValueError('unknown file extension: %s' % filepath)
    return img


import surface
import aux
import numpy as np
import warnings

from skimage import morphology
from scipy   import ndimage


BUGFIX_20180418A = aux.BUGFIX_DISABLED_CRITICAL
BUGFIX_20180605A = aux.BUGFIX_DISABLED
BUGFIX_20180614A = aux.BUGFIX_DISABLED


def rasterize_regions(regions, background_label=None, radius=3):
    borders = np.zeros(regions.shape, bool)
    background = np.zeros(regions.shape, bool)
    for i in xrange(regions.max() + 1):
        region_mask = (regions == i)
        interior = morphology.erosion(region_mask, morphology.disk(radius))
        border   = np.logical_and(region_mask, ~interior)
        borders[border] = True
        if i == background_label: background = interior.astype(bool)
    return borders, background


def render_regions_over_image(img, regions, background_label=None, bg=(0.6, 1, 0.6, 0.3), **kwargs):
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1), 'image has wrong dimensions'
    result = np.zeros((img.shape[0], img.shape[1], 3))
    for i in xrange(3): result[:, :, i] = img
    borders, background = rasterize_regions(regions, background_label, **kwargs)
    borders = borders.astype(int)
    result[:, :, 1]  = (1 - borders) * result[:, :, 1] + borders
    for i in [0, 2]: result[:, :, i] *=  1 - borders
    for i in xrange(3): result[background, i] = bg[i] * bg[3] + result[background, i] * (1 - bg[3])
    return (255 * result).clip(0, 255).astype('uint8')


def normalize_image(img):
    if not np.allclose(img.std(), 0):
        img = img.clip(max([img.min(), img.mean() - img.std()]), min([img.max(), img.mean() + img.std()]))
    return img - img.min()


def fetch_image_from_data(data, normalize_img=True):
    img = data['g_raw']
    if normalize_img: img = normalize_image(img)
    return img


def render_superpixels(data, discarded_only=False, normalize_img=True, discarded_color=(0.3, 1, 0.3, 0.1), border_radius=2, override_img=None):
    img = fetch_image_from_data(data, normalize_img) if override_img is None else override_img
    regions = data['g_superpixels'] > 0 if discarded_only else data['g_superpixels']
    return render_regions_over_image(img / img.max(), regions, background_label=0, bg=discarded_color, radius=border_radius)


def rasterize_activity_regions(data, candidates_key='postprocessed_candidates'):
    regions = np.zeros(data['g_raw'].shape, 'uint16')
    for label, candidate in enumerate(data[candidates_key], start=1):
        mask = candidate.get_mask(data['g_superpixels'])
        regions[mask] = label
    return regions


def render_activity_regions(data, normalize_img=True, none_color=(0.3, 1, 0.3, 0.1), border_radius=2, **kwargs):
    img = fetch_image_from_data(data, normalize_img)
    regions = rasterize_activity_regions(data, **kwargs)
    return render_regions_over_image(img / img.max(), regions, background_label=0, bg=none_color, radius=border_radius)


def render_model_shapes_over_image(data, candidates_key='postprocessed_candidates', normalize_img=True, interior_alpha=0, border=5, override_img=None, colors='g'):
    is_legal = True        ## other values are currently not generated
    override_xmaps = None  ## other values are currently not required

    colormap = {'r': [0], 'g': [1], 'b': [2], 'y': [0,1], 't': [1,2]}
    assert (isinstance(colors, dict) and all(c in colormap.keys() for c in colors.values())) or colors in colormap.keys()

    if not aux.is_bugfix_enabled(BUGFIX_20180614A) or normalize_img:
        g = surface.Surface.create_from_image(fetch_image_from_data(data, normalize_img) if override_img is None else override_img)
    else:
        g = surface.Surface(data['g_raw'].shape)
        g.model = fetch_image_from_data(data, normalize_img) if override_img is None else override_img

    candidates = data[candidates_key]
    if is_legal == True: is_legal = lambda m: True
    x_maps = override_xmaps if override_xmaps is not None else g.get_map(normalized=False)
    if isinstance(x_maps, np.ndarray): x_maps = [x_maps] * len(candidates)
    assert len(x_maps) == len(candidates), 'number of `x_maps` and `candidates` mismatch'

    img = np.zeros((g.model.shape[0], g.model.shape[1], 3))
    for i in xrange(3): img[:, :, i] = g.model * g.mask
    border_erode_selem, border_dilat_selem = morphology.disk(border / 2), morphology.disk(border - border / 2)
    for candidate, x_map in zip(candidates, x_maps):
        model = candidate.result
        model_shape = (model.s(x_map) >= 0)
        interior = morphology.binary_erosion (model_shape, border_erode_selem)
        border   = morphology.binary_dilation(model_shape, border_dilat_selem) - interior
        if isinstance(colors, dict):
            if candidate not in colors: continue
            colorchannels = colormap[colors[candidate]]
        else:
            colorchannels = colormap[colors]
        for ch in xrange(3):
            img[interior.astype(bool), ch] += interior_alpha * (+1 if ch in colorchannels else -1)
            img[border  .astype(bool), ch]  = (1 if ch in colorchannels else 0)

    return (255 * img).clip(0, 255).astype('uint8')


def rasterize_objects(data, candidates_key, dilate=0):
    models = [candidate.result for candidate in data[candidates_key]]
    x_map = data['g'].get_map(normalized=False)

    # BUGFIX_20180418A is only relevant if dilate != 0
    if dilate == 0:
        dlation, erosion = None, None
    else:
        if aux.is_bugfix_enabled(BUGFIX_20180418A):
            dilation, erosion = (morphology.binary_dilation, morphology.binary_erosion)
        else:
            dilation, erosion = (morphology.dilation, morphology.erosion)

    for model in models:
        fg = (model.s(x_map) > 0)
        if dilate > 0:   fg = dilation(fg, morphology.disk( dilate))
        elif dilate < 0: fg =  erosion(fg, morphology.disk(-dilate))

        if fg.any(): yield fg


def rasterize_labels(data, candidates_key='postprocessed_candidates', merge_overlap_threshold=np.inf, dilate=0):
    objects = [obj for obj in rasterize_objects(data, candidates_key, dilate)]

    # First, we determine which objects overlap sufficiently
    merge_list = []
    merge_mask = [False] * len(objects)
    for i1, i2 in ((i1, i2) for i1, obj1 in enumerate(objects) for i2, obj2 in enumerate(objects[:i1])):
        obj1, obj2 = objects[i1], objects[i2]
        overlap = np.logical_and(obj1, obj2).sum() / (0. + min([obj1.sum(), obj2.sum()]))
        if overlap > merge_overlap_threshold:
            merge_list.append((i1, i2))  # i2 is always smaller than i1
            merge_mask[i1] = True

    # Next, we associate a (potentially non-unique) label to each object
    labels, obj_indices_by_label = range(1, 1 + len(objects)), {}
    for label, obj_idx in zip(labels, xrange(len(objects))): obj_indices_by_label[label] = [obj_idx]
    for merge_idx, merge_data in enumerate(merge_list):
        assert merge_data[1] < merge_data[0], 'inconsistent merge data'
        merge_label0  = len(objects) + 1 + merge_idx         # the new label for the merged objects
        merge_labels  = [labels[idx] for idx in merge_data]  # two labels of the objects to be merged
        if merge_labels[0] == merge_labels[1]: continue      # this can occur due to transitivity
        merge_indices = obj_indices_by_label[merge_labels[0]] + obj_indices_by_label[merge_labels[1]]
        for obj_idx in merge_indices: labels[obj_idx] = merge_label0
        obj_indices_by_label[merge_label0] = merge_indices
        for label in merge_labels: del obj_indices_by_label[label]
    del labels, merge_list, merge_mask

    # Finally, we merge the rasterized objects
    objects_by_label = dict((i[0], [objects[k] for k in i[1]]) for i in obj_indices_by_label.items())
    objects  = [(np.sum(same_label_objects, axis=0) > 0) for same_label_objects in objects_by_label.values()]
    result   = np.zeros(data['g'].model.shape, 'uint16')
    if len(objects) > 0:
        overlaps = (np.sum(objects, axis=0) > 1)
        for l, obj in enumerate(objects, 1): result[obj] = l
        background = (result == 0).copy()
        result[overlaps] = 0
        dist = ndimage.morphology.distance_transform_edt(result == 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            result = morphology.watershed(dist, result, mask=np.logical_not(background))

    # In rare cases it can happen that two or more objects overlap exactly, in which csae the above code
    # will eliminate both objects. We will fix this by checking for such occasions explicitly:
    if aux.is_bugfix_enabled(BUGFIX_20180605A):
        for obj in objects:
            obj_mask = ((result > 0) * 1 - (obj > 0) * 1 < 0)
            if obj_mask.any(): result[obj_mask] = result.max() + 1

    return result


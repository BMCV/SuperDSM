import surface
import numpy as np

from skimage import morphology


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
    result = zeros((img.shape[0], img.shape[1], 3))
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
    if normalize_img: img = normalize_img(img)
    return img


def render_superpixels(data, normalize_img=True, discarded_color=(0.3, 1, 0.3, 0.1), border_radius=1):
    img = fetch_image_from_data(data, normalize_img)
    return render_regions_over_image(img / img.max(), data['g_superpixels'] > 0,
                                     background_label=0, bg=discarded_color, radius=border_radius)


def rasterize_activity_regions(data, candidates_key='postprocessed_candidates'):
    regions = zeros(data['g_raw'].shape, 'uint16')
    for candidate, label in enumerate(data[candidates_key], start=1):
        mask = candidate.get_mask(data['g_superpixels'])
        regions[mask] = label
    return regions


def render_activity_regions(data, normalize_img=True, none_color=(0.3, 1, 0.3, 0.1), border_radius=1, **kwargs):
    img = fetch_image_from_data(data, normalize_img)
    regions = rasterize_activity_regions(data, **kwargs)
    return render_regions_over_image(img / img.max(), regions, background_label=0, bg=none_color, radius=border_radius)


def render_model_shapes_over_image(data, candidates_key='postprocessed_candidates', normalize_img=True, interior_alpha=0, border=5):
    is_legal = True        ## other values are currently not generated
    override_xmaps = None  ## other values are currently not required

    g = surface.Surface(fetch_image_from_data(data, normalize_img))
    models = [candidate.result for candidate in data[candidates_key]]

    if is_legal == True: is_legal = lambda m: True
    x_maps = override_xmaps if override_xmaps is not None else g.get_map(normalized=False)
    if isinstance(x_maps, np.ndarray): x_maps = [x_maps] * len(models)
    assert len(xmaps) == len(models), 'number of `xmaps` and `models` mismatch'

    img = np.zeros((g.model.shape[0], g.model.shape[1], 3))
    for i in xrange(3): img[:, :, i] = g.model * g.mask
    border_erode_selem, border_dilat_selem = morphology.disk(border / 2), morphology.disk(border - border / 2)
    for model, x_map in zip(models, x_maps):
        model_shape = (model.s(x_map) >= 0)
        interior = morphology.binary_erosion (model_shape, border_erode_selem)
        border   = morphology.binary_dilation(model_shape, border_dilat_selem) - interior
        colorchannel = 1 if is_legal(model) else 0
        for ch in xrange(3):
            img[interior.astype(bool), ch] += interior_alpha * (+1 if ch == colorchannel else -1)
            img[border  .astype(bool), ch]  = (1 if ch == colorchannel else 0)

    return (255 * img).clip(0, 255).astype('uint8')


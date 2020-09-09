import gocell.surface
import gocell.aux

import numpy as np
import warnings, math

from skimage import morphology
from scipy   import ndimage

import skimage.draw
import matplotlib.pyplot as plt


def draw_line(p1, p2, thickness, shape):
    assert thickness >= 1
    threshold = (thickness + 1) / 2
    if np.allclose(threshold, round(threshold)):
        box = np.array((np.min((p1, p2), axis=0), np.max((p1, p2), axis=0)))
        n = math.ceil(threshold) - 1
        box[0] -= n
        box[1] += n
        box = box.clip(0, np.subtract(shape, 1))
        buf = np.zeros(1 + box[1] - box[0])
        p1  = p1 - box[0]
        p2  = p2 - box[0]
        rr, cc = skimage.draw.line(*p1, *p2)
        buf[rr, cc] = 1
        buf = ndimage.distance_transform_edt(buf == 0) < threshold
        result = np.zeros(shape)
        result[box[0,0] : box[1,0] + 1, box[0,1] : box[1,1] + 1] = buf
        return result
    else:
        thickness1 = 2 * int((thickness + 1) // 2) - 1
        thickness2 = thickness1 + 2
        buf1 = draw_line(p1, p2, thickness1, shape)
        buf2 = draw_line(p1, p2, thickness2, shape)
        return (buf2 * (thickness - thickness1) / (thickness2 - thickness1) + buf1).clip(0, 1)


def render_adjacencies(data, normalize_img=True, edge_thickness=3, endpoint_radius=5, endpoint_edge_thickness=2,
                       edge_color=(1,0,0), endpoint_color=(1,0,0), endpoint_edge_color=(0,0,0), override_img=None):
    if override_img is not None:
        assert override_img.ndim == 3 and override_img.shape[2] >= 3
        img = override_img[:, :, :3].copy()
        if (img > 1).any(): img /= 255
    else:
        img = np.dstack([_fetch_image_from_data(data, normalize_img)] * 3)
        img = img / img.max()
    lines = data['adjacencies'].get_edge_lines()
    shape = img.shape[:2]
    for endpoint in data['seeds']:
        perim_mask  = skimage.draw.circle(*endpoint, endpoint_radius + endpoint_edge_thickness, shape)
        for i in range(3):
            img[:,:,i][ perim_mask] = endpoint_edge_color[i]
    for line in lines:
        line_buf  = draw_line(line[0], line[1], edge_thickness, shape=shape)
        line_mask = (line_buf > 0)
        line_vals = line_buf[line_mask]
        for i in range(3): img[:, :, i][line_mask] = (line_vals) * edge_color[i]
    for endpoint in data['seeds']:
        circle_mask = skimage.draw.circle(*endpoint, endpoint_radius, shape)
        for i in range(3):
            img[:,:,i][circle_mask] = endpoint_color[i]
    return (255 * img).clip(0, 255).astype('uint8')


def render_ymap(data, clim=None):
    y = data['y'] if isinstance(data, dict) else data
    if clim is None: clim = (-y.std(), +y.std())
    y  = y.clip(*clim)
    y -= y.min()
    y /= y.max()
    return plt.cm.bwr(y)


def normalize_image(img):
    if not np.allclose(img.std(), 0):
        img = img.clip(max([img.min(), img.mean() - img.std()]), min([img.max(), img.mean() + img.std()]))
    return img - img.min()


def _fetch_image_from_data(data, normalize_img=True):
    img = data['g_raw']
    if normalize_img: img = normalize_image(img)
    return img


def render_atoms(data, discarded_only=False, normalize_img=True, discarded_color=(0.3, 1, 0.3, 0.1), border_radius=2, border_color=(0,1,0), override_img=None):
    img = _fetch_image_from_data(data, normalize_img) if override_img is None else override_img
    regions = data['g_atoms'] > 0 if discarded_only else data['g_atoms']
    return render_regions_over_image(img / img.max(), regions, background_label=0, bg=discarded_color, radius=border_radius, color=border_color)


def rasterize_regions(regions, background_label=None, radius=3):
    borders = np.zeros(regions.shape, bool)
    background = np.zeros(regions.shape, bool)
    for i in range(regions.max() + 1):
        region_mask = (regions == i)
        interior = morphology.erosion(region_mask, morphology.disk(radius))
        border   = np.logical_and(region_mask, ~interior)
        borders[border] = True
        if i == background_label: background = interior.astype(bool)
    return borders, background


def render_regions_over_image(img, regions, background_label=None, color=(0,1,0), bg=(0.6, 1, 0.6, 0.3), **kwargs):
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1,3)), f'image has wrong dimensions: {img.shape}'
    if img.ndim == 2 or img.shape[2] == 1:
        result = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3): result[:, :, i] = img
    else:
        result = img.copy()
    borders, background = rasterize_regions(regions, background_label, **kwargs)
    for i in range(3): result[:, :, i][borders] = color[i]
    #borders = borders.astype(int)
    #for i in range(3): result[:, :, i]  = color[i] * ((1 - borders) * result[:, :, 1] + borders)
    #for i in   [0, 2]: result[:, :, i] *=  1 - borders
    for i in range(3): result[background, i] = bg[i] * bg[3] + result[background, i] * (1 - bg[3])
    return (255 * result).clip(0, 255).astype('uint8')


def rasterize_activity_regions(data, candidates_key='postprocessed_candidates'):
    regions = np.zeros(data['g_raw'].shape, 'uint16')
    for label, candidate in enumerate(data[candidates_key], start=1):
        mask = candidate.get_mask(data['g_superpixels'])
        regions[mask] = label
    return regions


def render_activity_regions(data, normalize_img=True, none_color=(0.3, 1, 0.3, 0.1), border_radius=2, **kwargs):
    img = _fetch_image_from_data(data, normalize_img)
    regions = rasterize_activity_regions(data, **kwargs)
    return render_regions_over_image(img / img.max(), regions, background_label=0, bg=none_color, radius=border_radius)


COLORMAP = {'r': [0], 'g': [1], 'b': [2], 'y': [0,1], 't': [1,2]}


def render_model_shapes_over_image(data, candidates='postprocessed_candidates', normalize_img=True, interior_alpha=0, border=5, override_img=None, colors='g', labels=None):
    is_legal = True        ## other values are currently not generated
    override_xmaps = None  ## other values are currently not required

    assert (isinstance(colors, dict) and all(c in COLORMAP.keys() for c in colors.values())) or colors in COLORMAP.keys()

    if normalize_img:
        g = gocell.surface.Surface.create_from_image(_fetch_image_from_data(data, normalize_img) if override_img is None else override_img)
    else:
        g = gocell.surface.Surface(data['g_raw'].shape)
        g.model = _fetch_image_from_data(data, normalize_img) if override_img is None else override_img

    if isinstance(candidates, str): candidates = data[candidates]
    if is_legal == True: is_legal = lambda m: True

    img = np.zeros((g.model.shape[0], g.model.shape[1], 3))
    for i in range(3): img[:, :, i] = g.model * g.mask
    border_erode_selem, border_dilat_selem = morphology.disk(border / 2), morphology.disk(border - border / 2)
    merged_candidates = set()
    for candidate, foreground in zip(candidates, gocell.aux.render_candidate_foregrounds(g.model.shape, candidates)):
        if candidate in merged_candidates: continue
        merged_candidates |= {candidate}
        model_shape = foreground
        if labels is not None:
            label = np.bincount(labels[model_shape]).argmax()
            for candidate1, foreground1 in zip(candidates, gocell.aux.render_candidate_foregrounds(g.model.shape, candidates)):
                if candidate1 in merged_candidates: continue
                model1_shape = foreground1
                label1 = np.bincount(labels[model1_shape]).argmax()
                if label != label1: continue
                merged_candidates |= {candidate1}
                model_shape[model1_shape] = True
        interior = morphology.binary_erosion (model_shape, border_erode_selem)
        border   = morphology.binary_dilation(model_shape, border_dilat_selem) ^ interior
        if isinstance(colors, dict):
            if candidate not in colors: continue
            colorchannels = COLORMAP[colors[candidate]]
        else:
            colorchannels = COLORMAP[colors]
        for ch in range(3):
            img[interior.astype(bool), ch] += interior_alpha * (+1 if ch in colorchannels else -1)
            img[border  .astype(bool), ch]  = (1 if ch in colorchannels else 0)

    return (255 * img).clip(0, 255).astype('uint8')


def render_postprocessed_result(data, postprocessed_candidates='postprocessed_candidates', seg_border=5, color_accepted='g', color_discarded='r'):
    if isinstance(postprocessed_candidates, str): postprocessed_candidates = data[postprocessed_candidates]
    candidates, colors = [], {}
    for candidate in data['cover'].solution:
        postprocessed_candidate = [c for c in postprocessed_candidates if c.original is candidate]
        if len(postprocessed_candidate) > 0:
            candidates.append(postprocessed_candidate[0])
            colors[postprocessed_candidate[0]] = color_accepted
        else:
            candidates.append(candidate)
            colors[candidate] = color_discarded
    return gocell.render.render_model_shapes_over_image(data, candidates=candidates, border=seg_border, colors=colors)


def render_result_over_image(data, candidates_key='postprocessed_candidates', merge_overlap_threshold=np.inf, normalize_img=True, border=6, override_img=None, colors='g', gt_seg=None, gt_radius=8, gt_color='r'):
    assert (isinstance(colors, dict) and all(c in COLORMAP.keys() for c in colors.values())) or colors in COLORMAP.keys()
    assert gt_color in COLORMAP.keys()

    im_seg  = np.dstack([_fetch_image_from_data(data, normalize_img=normalize_img) if override_img is None else override_img] * 3).copy()
    im_seg /= im_seg.max()
    seg_objects = rasterize_labels(data, merge_overlap_threshold=merge_overlap_threshold)
    for l in set(seg_objects.flatten()) - {0}:
        seg_obj = (seg_objects == l)
        seg_bnd = np.logical_xor(morphology.binary_erosion(seg_obj, morphology.disk(border / 2)), morphology.binary_dilation(seg_obj, morphology.disk(border)))
        if isinstance(colors, dict):
            if candidate not in colors: continue
            colorchannels = COLORMAP[colors[candidate]]
        else:
            colorchannels = COLORMAP[colors]
        for i in range(3): im_seg[seg_bnd, i] = (1 if i in colorchannels else 0)
    if gt_seg is not None:
        xmap = np.indices(im_seg.shape[:2])
        for l in set(gt_seg.flatten()) - {0}:
            gt_obj = (gt_seg == l)
            gt_obj_center = np.asarray(ndimage.center_of_mass(gt_obj))
            gt_obj_dist   = np.linalg.norm(xmap - gt_obj_center[:,None,None], axis=0)
            for i in range(3): im_seg[gt_obj_dist <= gt_radius, i] = (1 if i in COLORMAP[gt_color] else 0)
    return (255 * im_seg).round().clip(0, 255).astype('uint8')


def rasterize_objects(data, candidates, dilate=0):
    if isinstance(candidates, str): candidates = [c for c in data[candidates]]

    if dilate == 0:
        dlation, erosion = None, None
    else:
        dilation, erosion = (morphology.binary_dilation, morphology.binary_erosion)

    for foreground in gocell.aux.render_candidate_foregrounds(data['g_raw'].shape, candidates):
        if dilate > 0:   foreground = dilation(foreground, morphology.disk( dilate))
        elif dilate < 0: foreground =  erosion(foreground, morphology.disk(-dilate))
        if foreground.any(): yield foreground.copy()


def rasterize_labels(data, candidates='postprocessed_candidates', merge_overlap_threshold=np.inf, dilate=0, background_label=0):
    assert background_label <= 0
    objects = [obj for obj in rasterize_objects(data, candidates, dilate)]

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
    labels, obj_indices_by_label = list(range(1, 1 + len(objects))), {}
    for label, obj_idx in zip(labels, range(len(objects))): obj_indices_by_label[label] = [obj_idx]
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
    result   = np.zeros(data['g_raw'].shape, 'uint16')
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
    for obj in objects:
        obj_mask = ((result > 0) * 1 - (obj > 0) * 1 < 0)
        if obj_mask.any(): result[obj_mask] = result.max() + 1

    result[result == 0] = background_label
    return result


def shuffle_labels(labels, bg_label=None, seed=None):
    label_values0 = frozenset(labels.flatten())
    if bg_label is not None: label_values0 -= {bg_label}
    label_values0 = list(label_values0)
    if seed is not None: np.random.seed(seed)
    label_values1 = np.asarray(label_values0).copy()
    np.random.shuffle(label_values1)
    label_map = dict(zip(label_values0, label_values1))
    result = np.zeros_like(labels)
    for l in label_map.keys():
        cc = (labels == l)
        result[cc] = label_map[l]
    return result


def colorize_labels(labels, bg_label=0, cmap='gist_rainbow', bg_color=(0,0,0), shuffle=None):
    if shuffle is not None:
        labels = shuffle_labels(labels, bg_label=bg_label, seed=shuffle)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    img = cmap((labels - labels.min()) / float(labels.max() - labels.min()))
    if img.shape[2] > 3: img = img[:,:,:3]
    if bg_label is not None:
        bg = (labels == bg_label)
        img[bg] = np.asarray(bg_color)[None, None, :]
    return img


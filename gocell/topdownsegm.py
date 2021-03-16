import gocell.config
import gocell.pipeline
import gocell.aux
import gocell.candidates
import gocell.atoms

import ray
import scipy.ndimage as ndi
import numpy as np
import skimage.segmentation as segm
import skimage.morphology as morph
import queue, contextlib, math, hashlib


def get_next_seed(region, where, score_func, connectivity=4):
    if   connectivity == 4: footprint = morph.disk(1)
    elif connectivity == 8: footprint = np.ones((3,3))
    else: raise ValeError(f'unknown connectivity: {connectivity}')
    mask  = np.logical_and(region.mask, where)
    image = region.model
    image_max = ndi.maximum_filter(image, footprint=footprint)
    max_mask  = np.logical_and(image_max == image, mask)
    if max_mask.any():
        maxima = ndi.label(max_mask)[0]
        maxima_labels = frozenset(maxima.reshape(-1)) - {0}
        scores = {max_label: score_func(maxima == max_label) for max_label in maxima_labels}
        label  = max(maxima_labels, key=scores.get)
        if scores[label] > -np.inf: return (maxima == label)
    return None


def watershed_split(region, *markers):
    markers_map = np.zeros(region.model.shape, int)
    for marker_label, marker in enumerate(markers, start=1):
        assert markers_map[marker] == 0
        markers_map[marker] = marker_label
    watershed = segm.watershed(region.model.max() - region.model.clip(0, np.inf), markers=markers_map, mask=region.mask)
    return [watershed == marker_label for marker_label in range(1, len(markers) + 1)]


def normalize_labels_map(labels, first_label=0, skip_labels=[]):
    result = np.zeros_like(labels)
    label_translation = {}
    next_label = first_label
    for old_label in sorted(np.unique(labels.reshape(-1))):
        if old_label in skip_labels: continue
        result[ labels == old_label] = next_label
        label_translation[old_label] = next_label
        next_label += 1
    return result, label_translation


def _hash_mask(mask):
    mask = mask.astype(np.uint8)
    return hashlib.sha1(mask).digest()


def get_cached_energy_rate_computer():
    cache = dict()
    def compute_energy_rate(candidate, region, atoms_map, mfcfg):
        _mf_kwargs = gocell.config.copy(mfcfg)
        bg_margin  = _mf_kwargs.pop('min_background_margin')
        mf_region  = candidate.get_modelfit_region(region, atoms_map, min_background_margin=bg_margin)
        mf_region_hash = _hash_mask(mf_region.mask)
        cache_hit  = cache.get(mf_region_hash, None)
        if cache_hit is None:
            if (mf_region.model[mf_region.mask] > 0).all() or (mf_region.model[mf_region.mask] < 0).all():
                energy = 0
            else:
                with contextlib.redirect_stdout(None):
                    J, model, status = gocell.candidates._modelfit(mf_region, smooth_mat_allocation_lock=None, **_mf_kwargs)
                energy = J(model)
            cache_hit = energy / mf_region.mask.sum()
            cache[mf_region_hash] = cache_hit
        return cache_hit
    return compute_energy_rate


class TopDownSegmentation(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(TopDownSegmentation, self).__init__('top-down-segmentation',
                                                  inputs  = ['y', 'mfcfg'],
                                                  outputs = ['y_mask', 'g_atoms', 'adjacencies', 'seeds', 'clusters'])

    def process(self, input_data, cfg, out, log_root_dir):
        seed_connectivity = gocell.config.get_value(cfg, 'seed_connectivity', 4)
        min_region_radius = gocell.config.get_value(cfg, 'min_region_radius', 10)
        max_atom_energy_rate = gocell.config.get_value(cfg, 'max_atom_energy_rate', 0.05)
        min_energy_rate_improvement = gocell.config.get_value(cfg, 'min_energy_rate_improvement', 0.1)
        max_cluster_marker_irregularity = gocell.config.get_value(cfg, 'max_cluster_marker_irregularity', 0.2)

        mfcfg = gocell.config.copy(input_data['mfcfg'])
        mfcfg['smooth_amount'] = np.inf
        
        out.intermediate(f'Analyzing cluster markers...')
        y = gocell.surface.Surface.create_from_image(input_data['y'], normalize=False)
        fg_mask = (y.model > 0)
        fg_bd   = np.logical_xor(fg_mask, morph.binary_erosion(fg_mask, morph.disk(1)))
        y_mask  = np.ones(y.model.shape, bool)
        cluster_markers = ndi.label(fg_mask)[0]
        for cluster_marker_label in np.unique(cluster_markers):
            cluster_marker = (cluster_markers == cluster_marker_label)
            cluster_marker_irregularity = fg_bd[cluster_marker].sum() / cluster_marker.sum()
            if cluster_marker_irregularity > max_cluster_marker_irregularity:
                y_mask[cluster_marker] = False
                
        cluster_markers[~y_mask] = cluster_markers.min()
        cluster_markers = normalize_labels_map(cluster_markers, first_label=0)[0]
        out.write(f'Extracted {cluster_markers.max()} cluster markers')
        
        clusters  = segm.watershed(ndi.distance_transform_edt(cluster_markers == 0), markers=cluster_markers)
        atoms_map = np.full(y.model.shape, 0)
        atom_candidate_by_label = {}
        
        y_id = ray.put(y)
        mfcfg_id = ray.put(mfcfg)
        y_mask_id = ray.put(y_mask)
        clusters_id = ray.put(clusters)
        futures = [process_cluster.remote(clusters_id, cluster_label, y_id, y_mask_id, max_atom_energy_rate, min_region_radius, min_energy_rate_improvement, mfcfg_id, seed_connectivity) for cluster_label in frozenset(clusters.reshape(-1)) - {0}]
        for ret_idx, ret in enumerate(gocell.aux.get_ray_1by1(futures)):
            cluster_label, cluster_universe, cluster_atoms, cluster_atoms_map, cluster_max_energy_rate = ret
            cluster_label_offset = atoms_map.max()
            cluster = y.get_region(clusters == cluster_label)
            atoms_map[cluster.mask] = cluster_label_offset + cluster_atoms_map[cluster.mask]
            for atom_candidate in cluster_atoms:
                atom_candidate_by_label[cluster_label_offset + list(atom_candidate.footprint)[0]] = atom_candidate
            out.intermediate(f'Analyzing clusters... {ret_idx + 1} / {len(futures)}')
            
        atoms_map, label_translation = normalize_labels_map(atoms_map, first_label=1, skip_labels=[0])
        for old_label, atom_candidate in dict(atom_candidate_by_label).items():
            atom_candidate_by_label[label_translation[old_label]] = atom_candidate
        out.write(f'Extracted {atoms_map.max()} atoms')
        
        # Compute adjacencies graph
        atom_nodes  = [np.round(ndi.center_of_mass(atom_candidate_by_label[atom_label].seed)).astype(int) for atom_label in sorted(label_translation.values())]
        adjacencies = gocell.atoms.AtomAdjacencyGraph(atoms_map, clusters, fg_mask, atom_nodes, out)
        
        return {
            'y_mask': y_mask,
            'g_atoms': atoms_map,
            'adjacencies': adjacencies,
            'seeds': atom_nodes,
            'clusters': clusters
        }


@ray.remote
def process_cluster(*args, **kwargs):
    return _process_cluster_impl(*args, **kwargs)


def _process_cluster_impl(clusters, cluster_label, y, y_mask, max_atom_energy_rate, min_region_radius, min_energy_rate_improvement, mfcfg, seed_connectivity):
    min_region_size = 2 * math.pi * (min_region_radius ** 2)
    cluster = y.get_region(clusters == cluster_label)
    masked_cluster = cluster.get_region(y_mask)
    root_candidate = gocell.candidates.Candidate()
    root_candidate.footprint = frozenset([1])
    root_candidate.seed = get_next_seed(masked_cluster, cluster.model > 0, lambda loc: cluster.model[loc].max(), seed_connectivity)
    atoms_map = cluster.mask.astype(int) * list(root_candidate.footprint)[0]
    compute_energy_rate = get_cached_energy_rate_computer()

    leaf_candidates = []
    split_queue = queue.Queue()
    root_candidate.energy_rate = compute_energy_rate(root_candidate, masked_cluster, atoms_map, mfcfg)
    if root_candidate.energy_rate > max_atom_energy_rate:
        split_queue.put(root_candidate)
    else:
        leaf_candidates.append(root_candidate)

    seed_distances = ndi.distance_transform_edt(~root_candidate.seed)
    while not split_queue.empty():
        c0 = split_queue.get()
        c0_mask = c0.get_mask(atoms_map)

        c1 = gocell.candidates.Candidate()
        c2 = gocell.candidates.Candidate()
        c1.seed = c0.seed
        c2.seed = get_next_seed(masked_cluster, np.all((cluster.model > 0, c0_mask, seed_distances >= 1), axis=0), lambda loc: seed_distances[loc].max(), seed_connectivity)
        assert not np.logical_and(c1.seed, c2.seed).any()
        if c2.seed is None:
            leaf_candidates.append(c0)
            continue
        else:
            seed_distances = np.min([seed_distances, ndi.distance_transform_edt(~c2.seed)], axis=0)

        new_atom_label   = atoms_map.max() + 1
        c1_mask, c2_mask = watershed_split(cluster.get_region(c0_mask), c1.seed, c2.seed)
        if c1_mask.sum() < min_region_size or c2_mask.sum() < min_region_size:
            split_queue.put(c0) ## try again with different seed
            continue
            
        atoms_map_previous = atoms_map.copy()
        atoms_map[c2_mask] = new_atom_label
        c1.footprint = frozenset(c0.footprint)
        c2.footprint = frozenset([new_atom_label])
        assert c1_mask[cluster.mask].any() and not np.logical_and(~cluster.mask, c1_mask).any()
        assert c2_mask[cluster.mask].any() and not np.logical_and(~cluster.mask, c2_mask).any()

        for c in (c1, c2):
            c.energy_rate = compute_energy_rate(c, masked_cluster, atoms_map, mfcfg)
            
        energy_rate_improvement = 1 - max((c1.energy_rate, c2.energy_rate)) / c0.energy_rate
        if energy_rate_improvement < min_energy_rate_improvement:
            split_queue.put(c0) ## try again with different seed
            atoms_map = atoms_map_previous
        else:
            for c in (c1, c2):
                if c.energy_rate > max_atom_energy_rate:
                    split_queue.put(c)
                else:
                    leaf_candidates.append(c)

    root_candidate.footprint = frozenset(atoms_map.reshape(-1)) - {0}
    assert frozenset([list(c.footprint)[0] for c in leaf_candidates]) == root_candidate.footprint
    max_energy_rate = max((c.energy_rate for c in leaf_candidates))
    return cluster_label, root_candidate, leaf_candidates, atoms_map, max_energy_rate
    

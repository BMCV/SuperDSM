import gocell.aux
import gocell.config
import gocell.pipeline

import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndi
import scipy.special
import warnings


class AtomicStage(gocell.pipeline.Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(AtomicStage, self).__init__('atoms',
                                          inputs  = ['y', 'foreground_labels', 'seeds'],
                                          outputs = ['g_clusters', 'g_atoms', 'adjacencies'])

    def process(self, input_data, cfg, out, log_root_dir):
        shape = input_data['y'].shape
        bg_mask = (input_data['foreground_labels'] == 0)
        split_clusters = gocell.config.get_value(cfg, 'split_clusters', 0)

        # Apply watershed transform using image intensities
        cc_distances = ndi.distance_transform_edt(bg_mask)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            g_clusters = morph.watershed(cc_distances)
        out.write('Clusters: %d' % g_clusters.max())

        # Rasterize atom seeds
        g_atom_seeds = np.zeros(shape, 'uint16')
        for seed_idx, seed in enumerate(input_data['seeds']):
            g_atom_seeds[tuple(seed)] = seed_idx + 1
        assert g_atom_seeds.max() == len(input_data['seeds'])

        g_atoms = np.zeros_like(g_clusters)
        for cluster_l in frozenset(g_clusters.reshape(-1)):
            cluster_mask = (g_clusters == cluster_l)

            # Apply watershed transform using image intensities
            distances = input_data['y'].max() - input_data['y'].clip(0, np.inf)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                g_cluster_atoms = morph.watershed(distances, g_atom_seeds * cluster_mask, mask=cluster_mask)
                for l in frozenset(g_cluster_atoms.reshape(-1)) - {0}:
                    cc = (g_cluster_atoms == l)
                    g_atoms[cc] = g_atom_seeds[cc].max()
            out.intermediate(f'Extracting atoms from cluster {cluster_l} / {g_clusters.max()}')
        out.write(f'Extracted atoms: {g_atoms.max()}')
        assert g_atoms.min() > 0

        # Compute adjacencies graph
        adjacencies = AtomAdjacencyGraph(g_atoms, g_clusters, ~bg_mask, out)

        # Optionally, split clusters
        if split_clusters > 0:
            fg_mask   = morph.binary_erosion(input_data['foreground_labels'] > 0, morph.disk(split_clusters))
            fg_labels = ndi.label(fg_mask)[0]
            removed_edges = 0
            for atom1 in list(adjacencies.atom_labels):
                seed1 = tuple(input_data['seeds'][atom1 - 1])
                for atom2 in list(adjacencies[atom1]):
                    seed2 = tuple(input_data['seeds'][atom2 - 1])
                    if fg_labels[seed1] != fg_labels[seed2]:
                       adjacencies.remove_adjacency(atom1, atom2)
                       removed_edges += 1
            out.write(f'Removed {removed_edges} edge(s)')

        # Estimate computational load
        comp_load_with_n_atoms = lambda n: sum(scipy.special.comb(n, k, exact=True) for k in range(1, n + 1))
        comp_load = sum(comp_load_with_n_atoms(len(adjacencies.get_atoms_in_cluster(cluster_label))) for cluster_label in adjacencies.cluster_labels)
        out.write(f'Computational load ≤ {comp_load:,}')

        return {
            'g_clusters':  g_clusters,
            'g_atoms':     g_atoms,
            'adjacencies': adjacencies
        }


class AtomAdjacencyGraph:

    def __init__(self, g_atoms, g_clusters, fg_mask, out=None):
        out = gocell.aux.get_output(out)
        self._adjacencies, se = {atom_label: set() for atom_label in range(1, g_atoms.max() + 1)}, morph.disk(1)
        self._atoms_by_cluster, self._cluster_by_atom = {}, {}
        for l0 in range(1, g_atoms.max() + 1):
            cc = (g_atoms == l0)
            cluster_label = g_clusters[cc][0]
            cluster_mask  = np.logical_and(fg_mask, g_clusters == cluster_label)
            cc_dilated = np.logical_and(morph.binary_dilation(cc, se), np.logical_not(cc))
            cc_dilated = np.logical_and(cc_dilated, cluster_mask)
            if cluster_label not in self._atoms_by_cluster:
                self._atoms_by_cluster[cluster_label] = set()
            adjacencies = set(g_atoms[cc_dilated].flatten()) - {0, l0}
            self._adjacencies[l0] |= adjacencies
            for l1 in adjacencies:
                self._adjacencies[l1] |= {l0}
            self._cluster_by_atom[l0] = cluster_label
            self._atoms_by_cluster[cluster_label] |= {l0}

            out.intermediate('Processed atom %d / %d' % (l0, g_atoms.max()))
        out.write('Computed atom adjacencies')
        assert self.is_symmetric()
    
    def __getitem__(self, atom_label):
        return self._adjacencies[atom_label]

    def remove_adjacency(self, atom_label1, atom_label2):
        self._adjacencies[atom_label1].remove(atom_label2)
        self._adjacencies[atom_label2].remove(atom_label1)
        self._update_clusters(atom_label1)
        self._update_clusters(atom_label2)

    def _update_clusters(self, atom_label):
        old_cluster_label = self._cluster_by_atom[atom_label]
        if len(self[atom_label]) == 0 and len(self._atoms_by_cluster[old_cluster_label]) > 1:
            new_cluster_label = max(self.cluster_labels) + 1
            self._cluster_by_atom[atom_label] = new_cluster_label
            self._atoms_by_cluster[new_cluster_label]  = {atom_label}
            self._atoms_by_cluster[old_cluster_label] -= {atom_label}
    
    def get_cluster_label(self, atom_label):
        return self._cluster_by_atom[atom_label]
    
    def get_atoms_in_cluster(self, cluster_label):
        return self._atoms_by_cluster[cluster_label]
    
    @property
    def cluster_labels(self): return frozenset(self._atoms_by_cluster.keys())
    
    @property
    def atom_labels(self): return frozenset(self._cluster_by_atom.keys())
    
    ACCEPT_ALL_ATOMS = lambda atom_label: True
    
    def get_edge_lines(self, data, accept=ACCEPT_ALL_ATOMS):
        lines = []
        for l in frozenset(data['g_atoms'].reshape(-1)):
            seed_l = data['seeds'][l - 1]
            if not accept(l): continue
            for k in self[l]:
                seed_k = data['seeds'][k - 1]
                if not accept(k): continue
                lines.append([seed_l, seed_k])
        return lines

    @property
    def max_degree(self):
        return max(self.get_atom_degree(atom_label) for atom_label in self.atom_labels)

    def get_atom_degree(self, atom_label):
        return len(self[atom_label])

#    def get_search_depth(self, atom_label):
#        visited_atoms = {atom_label}
#        next_atoms    = [(neighbor, 1) for neighbor in self[atom_label]]
#        max_depth     = 0
#        while len(next_atoms) > 0:
#            current_atom, current_depth = next_atoms[0]
#            next_atoms = next_atoms[1:]
#            visited_atoms |= {current_atom}
#            max_depth = max((max_depth, current_depth))
#            for neighbor in self[current_atom]:
#                if neighbor in visited_atoms or neighbor in (atom[0] for atom in next_atoms): continue
#                next_atoms.append((neighbor, current_depth + 1))
#        return max_depth

    def is_symmetric(self):
        for atom1 in self.atom_labels:
            if not all(atom1 in self[atom2] for atom2 in self[atom1]):
                return False
        return True


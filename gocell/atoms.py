import gocell.aux
import gocell.config
import gocell.pipeline

import skimage.morphology as morph


class AtomStage(gocell.pipeline.Stage)

    def __init__(self):
        super(PreprocessingStage, self).__init__('atoms',
                                                 inputs  = ['y', 'foreground_labels', 'seeds'],
                                                 outputs = ['g_clusters', 'g_atoms', 'adjacencies'])

    def process(self, input_data, cfg, out, log_root_dir):
        shape = data['y'].shape

        # Apply watershed transform using image intensities
        cc_distances = ndi.distance_transform_edt(data['foreground_labels'] == 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            g_clusters = morph.watershed(cc_distances)
        out.write('Clusters: %d' % g_clusters.max())

        # Rasterize atom seeds
        g_atom_seeds = np.zeros(shape, 'uint16')
        for seed_idx, seed in enumerate(data['seeds']):
            g_atom_seeds[tuple(seed)] = seed_idx + 1

        g_atoms = zeros_like(g_clusters)
        for cluster_l in frozenset(g_clusters.reshape(-1)):
            cluster_mask = (data['g_clusters'] == cluster_l)

            # Apply watershed transform using image intensities
            distances = data['y'].max() - data['y'].clip(0, np.inf)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                g_cluster_atoms = morph.watershed(distances, g_atom_seeds * cluster_mask, mask=cluster_mask)
                for l in frozenset(g_cluster_atoms.reshape(-1)) - {0}:
                    cc = (g_cluster_atoms == l)
                    g_atoms[cc] = g_atom_seeds[cc].max()
            out.intermediate(f'Processing cluster: {cluster_l} / {g_clusters.max()}')
        out.write('Atoms: %d' % g_atoms.max())

        adjacencies = AtomAdjacencyGraph(g_atoms, g_clusters, out)
        return {
            'g_clusters':  g_clusters,
            'g_atoms':     g_atoms,
            'adjacencies': adjacencies
        }


class AtomAdjacencyGraph:

    def __init__(self, g_atoms, g_clusters, out=None):
        out = gocell.aux.Output.get(out)
        self._adjacencies, se = {}, morph.disk(1)
        self._atoms_by_cluster, self._cluster_by_atom = {}, {}
        for l0 in range(1, g_atoms.max() + 1):
            cc = (g_atoms == l0)
            cluster_label =  g_clusters[cc][0]
            cluster_mask  = (g_clusters == cluster_label)
            cc_dilated = np.logical_and(morph.binary_dilation(cc, se), np.logical_not(cc))
            cc_dilated = np.logical_and(cc_dilated, cluster_mask)
            if cluster_label not in self._atoms_by_cluster:
                self._atoms_by_cluster[cluster_label] = set()
            self._adjacencies[l0] = set(g_atoms[cc_dilated].flatten()) - {0, l0}
            self._cluster_by_atom[l0] = cluster_label
            self._atoms_by_cluster[cluster_label] |= {l0}

            out.intermediate('Processed atom %d / %d' % (l0, g_atoms.max()))
        out.write('Computed atom adjacencies')
    
    def __getitem__(self, atom_label):
        return self._adjacencies[atom_label]
    
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
            for k in adjacencies[l]:
                seed_k = data['seeds'][k - 1]
                if not accept(k): continue
                lines.append([seed_l, seed_k])
        return lines


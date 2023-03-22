from .output import get_output

import numpy as np
import skimage.morphology as morph
import skimage.segmentation


class AtomAdjacencyGraph:
    """Graph representation of the adjacencies of atomic image regions.

    This corresponds to the adjacency graph :math:`\\mathcal G` as defined in :ref:`pipeline_theory_c2freganal`.

    :param atoms: Integer-valued image representing the universe of atomic image regions. Each atomic image region has a unique label, which is the integer value.
    :param clusters: Integer-valued image representing the regions of possibly clustered obejcts. Each region has a unique label, which is the integer value.
    :param fg_mask: Binary image corresponding to a rough representation of the image foreground. This means that an image point :math:`x \\in \\Omega` is ``True`` if :math:`Y_\\omega|_{\\omega=\\{x\\}} > 0` and ``False`` otherwise.
    :param seeds: The seed points which were used to determine the atomic image regions, represented by a list of tuples of coordinates.
    :param out: An output object obtained via :py:meth:`~superdsm.output.get_output`, or ``None`` if the default output should be used.
    """

    def __init__(self, atoms, clusters, fg_mask, seeds, out=None):
        out = get_output(out)
        self._adjacencies, se = {atom_label: set() for atom_label in range(1, atoms.max() + 1)}, morph.disk(1)
        self._atoms_by_cluster, self._cluster_by_atom = {}, {}
        self._seeds = seeds
        for l0 in range(1, atoms.max() + 1):
            cc = (atoms == l0)
            cluster_label = clusters[cc][0]
            cluster_mask  = np.logical_and(fg_mask, clusters == cluster_label)
            cc_dilated = np.logical_and(morph.binary_dilation(cc, se), np.logical_not(cc))
            cc_dilated = np.logical_and(cc_dilated, cluster_mask)
            if cluster_label not in self._atoms_by_cluster:
                self._atoms_by_cluster[cluster_label] = set()
            adjacencies = set(atoms[cc_dilated].flatten()) - {0, l0}
            self._adjacencies[l0] |= adjacencies
            for l1 in adjacencies:
                self._adjacencies[l1] |= {l0}
            self._cluster_by_atom[l0] = cluster_label
            self._atoms_by_cluster[cluster_label] |= {l0}

            out.intermediate('Processed atom %d / %d' % (l0, atoms.max()))
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

    def get_seed(self, atom_label):
        return self._seeds[atom_label - 1]
    
    def get_edge_lines(self, accept=ACCEPT_ALL_ATOMS):
        lines = []
        for l in self.atom_labels:
            seed_l = self.get_seed(l)
            if not accept(l): continue
            for k in self[l]:
                seed_k = self.get_seed(k)
                if not accept(k): continue
                lines.append([seed_l, seed_k])
        return lines

    @property
    def max_degree(self):
        return max(self.get_atom_degree(atom_label) for atom_label in self.atom_labels)

    def get_atom_degree(self, atom_label):
        return len(self[atom_label])

    def is_symmetric(self):
        for atom1 in self.atom_labels:
            if not all(atom1 in self[atom2] for atom2 in self[atom1]):
                return False
        return True


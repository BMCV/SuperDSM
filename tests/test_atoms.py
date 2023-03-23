import unittest
import numpy as np
import superdsm.atoms

from . import testsuite


class atoms(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        with testsuite.SilentOutputContext() as out:
            self.atoms = np.array([[1, 1, 2, 4],
                                   [1, 3, 2, 4],
                                   [3, 3, 3, 4]])
            self.clusters = np.array([[1, 1, 2, 2],
                                      [1, 2, 2, 2],
                                      [2, 2, 2, 2]])
            self.fg_mask = np.array([[True, False, True, False],
                                     [True, False, True,  True],
                                     [True,  True, True,  True]])
            self.seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
            self.adj = superdsm.atoms.AtomAdjacencyGraph(self.atoms, self.clusters, self.fg_mask, self.seeds, out=out)

    def test_AtomAdjacencyGraph(self):
        self.assertEqual(self.adj[1], set( ))
        self.assertEqual(self.adj[2], {3, 4})
        self.assertEqual(self.adj[3], {2, 4})
        self.assertEqual(self.adj[4], {2, 3})

    def test_AtomAdjacencyGraph_atom_labels(self):
        self.assertEqual(self.adj.atom_labels, frozenset({1, 2, 3, 4}))

    def test_AtomAdjacencyGraph_cluster_labels(self):
        self.assertEqual(self.adj.cluster_labels, frozenset({1, 2}))

    def test_AtomAdjacencyGraph_get_atom_degree(self):
        self.assertEqual(self.adj.get_atom_degree(1), 0)
        self.assertEqual(self.adj.get_atom_degree(2), 2)
        self.assertEqual(self.adj.get_atom_degree(3), 2)
        self.assertEqual(self.adj.get_atom_degree(4), 2)

    def test_AtomAdjacencyGraph_get_atoms_in_cluster(self):
        self.assertEqual(self.adj.get_atoms_in_cluster(1), {1})
        self.assertEqual(self.adj.get_atoms_in_cluster(2), {2, 3, 4})

    def test_AtomAdjacencyGraph_get_cluster_label(self):
        self.assertEqual(self.adj.get_cluster_label(1), 1)
        self.assertEqual(self.adj.get_cluster_label(2), 2)
        self.assertEqual(self.adj.get_cluster_label(3), 2)
        self.assertEqual(self.adj.get_cluster_label(4), 2)

    def test_AtomAdjacencyGraph_get_edge_lines(self):
        self.assertEqual(self.adj.get_edge_lines(), [((0, 2), (2, 1)), ((0, 2), (1, 3)), ((2, 1), (1, 3))])
        self.assertEqual(self.adj.get_edge_lines(lambda i: i != 4), [((0, 2), (2, 1))])
        self.assertEqual(self.adj.get_edge_lines(lambda i: i != 4, reduce=False), [((0, 2), (2, 1)), ((2, 1), (0, 2))])

    def test_AtomAdjacencyGraph_get_seed(self):
        with testsuite.SilentOutputContext() as out:
            adj2 = superdsm.atoms.AtomAdjacencyGraph(self.atoms, self.clusters, self.fg_mask, self.seeds[::-1], out=out)
        for adj in (self.adj, adj2):
            self.assertEqual(adj.get_seed(1), (0, 0))
            self.assertEqual(adj.get_seed(2), (0, 2))
            self.assertEqual(adj.get_seed(3), (2, 1))
            self.assertEqual(adj.get_seed(4), (1, 3))

    def test_AtomAdjacencyGraph_max_degree(self):
        self.assertEqual(self.adj.max_degree, 2)


if __name__ == '__main__':
    unittest.main()

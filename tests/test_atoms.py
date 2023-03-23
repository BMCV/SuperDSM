import unittest
import numpy as np
import superdsm.atoms

from . import testsuite


class atoms(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        with testsuite.SilentOutputContext() as out:
            atoms = np.array([[1, 1, 2, 4],
                                [1, 3, 2, 4],
                                [3, 3, 3, 4]])
            clusters = np.array([[1, 1, 2, 2],
                                    [1, 2, 2, 2],
                                    [2, 2, 2, 2]])
            fg_mask = np.array([[True, False, True],
                                [True, False, True],
                                [True,  True, True]])
            seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
            adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, out=out)

    def test_AtomAdjacencyGraph(self):
        pass
        #adj[1]
        #adj[2]
        #adj[3]
        #adj[4]


if __name__ == '__main__':
    unittest.main()

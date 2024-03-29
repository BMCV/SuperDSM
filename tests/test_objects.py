import unittest
import numpy as np
import superdsm.objects
import superdsm.image

from . import testsuite


class objects(unittest.TestCase):

    def test_BaseObject_fill_foreground(self):
        obj = superdsm.objects.BaseObject()
        obj.fg_fragment = np.array([[False,  True],
                                    [ True,  True],
                                    [ True, False]])
        obj.fg_offset = (1, 2)
        actual = np.zeros((4, 5), bool)
        obj.fill_foreground(actual)
        expected = np.array([[False, False, False, False, False],
                             [False, False, False,  True, False],
                             [False, False,  True,  True, False],
                             [False, False,  True, False, False]])
        np.testing.assert_allclose(actual, expected)

    def test_Object_get_mask(self):
        atoms = np.array([[1, 1, 2],
                          [1, 3, 2],
                          [3, 3, 3]])
        obj = superdsm.objects.Object()
        obj.footprint = set([2, 3])
        actual = obj.get_mask(atoms)
        expected = np.array([[False, False,  True],
                             [False,  True,  True],
                             [ True,  True,  True]])
        np.testing.assert_allclose(actual, expected)

    def test_extract_foreground_fragment(self):
        mask = np.array([[False, False, False, False, False],
                         [False, False, False,  True, False],
                         [False, False,  True,  True, False],
                         [False, False,  True, False, False]])
        actual_offset, actual_fragment = superdsm.objects.extract_foreground_fragment(mask)
        expected_offset = np.array([1, 2])
        expected_fragment = np.array([[False,  True],
                                      [ True,  True],
                                      [ True, False]])
        np.testing.assert_allclose(actual_offset, expected_offset)
        np.testing.assert_allclose(actual_fragment, expected_fragment)

    def test_get_cvxprog_region(self):
        y_data = np.array([[-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, +1, -1, -1, -1],
                           [-1, +1, -1, -1, +1],
                           [-1, +1, -1, -1, +1]])
        atoms  = np.array([[ 1,  1,  1,  1,  1],
                           [ 1,  1,  1,  1,  1],
                           [ 1,  1,  1,  1,  2],
                           [ 1,  1,  1,  2,  2],
                           [ 1,  1,  1,  2,  2],
                           [ 1,  1,  1,  2,  2]])
        obj = superdsm.objects.Object()
        obj.footprint = set([1])
        y = superdsm.image.Image(y_data)
        region = obj.get_cvxprog_region(y, atoms, background_margin=2)
        expected = np.array([[False, False, False, False, False],
                             [False,  True, False, False, False],
                             [ True,  True,  True, False, False],
                             [ True,  True,  True, False, False],
                             [ True,  True,  True, False, False],
                             [ True,  True,  True, False, False]])
        np.testing.assert_allclose(region.mask, expected)
        np.testing.assert_allclose(region.model, y_data)


if __name__ == '__main__':
    unittest.main()

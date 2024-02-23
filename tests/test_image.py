import unittest
import numpy as np
import superdsm.image

from . import testsuite


class image(unittest.TestCase):

    def test_get_pixel_map(self):
        actual1 = superdsm.image.get_pixel_map((5, 5))
        actual2 = superdsm.image.get_pixel_map((5, 5), normalized=True)
        expected = \
         np.array([[[0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1.],
                    [2., 2., 2., 2., 2.],
                    [3., 3., 3., 3., 3.],
                    [4., 4., 4., 4., 4.]],

                   [[0., 1., 2., 3., 4.],
                    [0., 1., 2., 3., 4.],
                    [0., 1., 2., 3., 4.],
                    [0., 1., 2., 3., 4.],
                    [0., 1., 2., 3., 4.]]])
        np.testing.assert_allclose(actual1, expected)
        np.testing.assert_allclose(actual2, expected / 4)

    def test_bbox(self):
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 0, 0]])
        actual1 = superdsm.image.bbox(mask.astype(bool))
        actual2 = superdsm.image.bbox(mask.astype(bool), include_end=True)
        expected1 = (np.array([[1, 4], [2, 4]]), (slice(1, 4, None), slice(2, 4, None)))
        expected2 = (np.array([[1, 3], [2, 3]]), (slice(1, 3, None), slice(2, 3, None)))
        np.testing.assert_allclose(actual1[0], expected1[0])
        np.testing.assert_allclose(actual2[0], expected2[0])
        self.assertEqual(actual1[1], expected1[1])
        self.assertEqual(actual2[1], expected2[1])


if __name__ == '__main__':
    unittest.main()

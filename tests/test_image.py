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


if __name__ == '__main__':
    unittest.main()

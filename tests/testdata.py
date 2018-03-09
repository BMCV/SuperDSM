import sys
import numpy as np

from math import sqrt

sys.path.append('..')

import gocell.surface
import gocell.modelfit_base


def create_surface(seed=0):
    np.random.seed(seed)
    g = gocell.surface.Surface((200, 200))
    groundtruth = gocell.modelfit_base.PolynomialModel.create_ellipsoid(np.array((0.5, 0.5)), 0.2, 0.4)
    g.model[groundtruth.s(g.get_map()) < 0] = 1
    return g, groundtruth


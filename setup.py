#!/usr/bin/env python

from distutils.core import setup

import segmetrics


setup(
    name = 'SuperDSM',
    version = superdsm.VERSION,
    description = 'SuperDSM is a globally optimal segmentation method based on superadditivity and deformable shape models for cell nuclei in fluorescence microscopy images and beyond.',
    author = 'Leonid Kostrykin',
    author_email = 'leonid.kostrykin@bioquant.uni-heidelberg.de',
    url = 'https://kostrykin.com',
    license = 'MIT',
    packages = ['superdsm'],
)

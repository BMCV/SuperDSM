#!/usr/bin/env python

#import os
#try:
#    os.symlink(f'../../sparse_dot/sparse_dot_mkl/{filename}', f'superdsm/sparse_dot_mkl/{filename}')
#except FileExistsError:
#    pass

from distutils.core import setup

import superdsm


setup(
    name = 'SuperDSM',
    version = superdsm.__version__,
    description = 'SuperDSM is a globally optimal segmentation method based on superadditivity and deformable shape models for cell nuclei in fluorescence microscopy images and beyond.',
    author = 'Leonid Kostrykin',
    author_email = 'leonid.kostrykin@bioquant.uni-heidelberg.de',
    url = 'https://kostrykin.com',
    license = 'MIT',
    packages = ['superdsm', 'superdsm._libs', 'superdsm._libs.sparse_dot_mkl'],
)

#!/usr/bin/env python

from distutils.core import setup

# We avoid loading the superdsm module here because we don't want to load the dependencies when only building the documentation (libmkl_rt cannot be loaded by readthedocs).
#with open('superdsm/version.py', 'r') as fin:
#    version_locals = dict()
#    exec(fin.read(), None, version_locals)
#    locals().update(version_locals)
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

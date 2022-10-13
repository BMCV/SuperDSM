#!/usr/bin/env python

import os
import pathlib
import requests
import tempfile
import shutil

root_dir = pathlib.Path(os.path.realpath(__file__)).parent
data_dir = root_dir / 'data'

def load_data(url, datasets, **kwargs):
    with tempfile.NamedTemporaryFile('wb', suffix=pathlib.Path(url).suffix.lower()) as archive_file:
        print(f'Downloading archive: {url}')
        with requests.get(url, stream=True, **kwargs) as req:
            req.raise_for_status()
            for chunk in req.iter_content(chunk_size=10 * 1024 ** 2):
                archive_file.write(chunk)
        archive_file.flush()
        with tempfile.TemporaryDirectory() as archive_dirpath:
            print(f'Unpacking to {archive_dirpath}')
            shutil.unpack_archive(archive_file.name, archive_dirpath)
            src_root = pathlib.Path(archive_dirpath)
            for src, dst in datasets:
                print(f'Populating {data_dir / dst}')
                shutil.move(src_root / src, data_dir / dst)

load_data(
    'http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip',
    [
        ('Fluo-N2DH-GOWT1/01', 'GOWT1-1'),
        ('Fluo-N2DH-GOWT1/02', 'GOWT1-2'),
    ])

load_data(
    'https://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz',
    [
        ('data/images/dna-images/gnf', 'U2OS'),
        ('data/images/dna-images/ic100', 'NIH3T3'),
    ],
    verify=False)

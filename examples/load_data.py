#!/usr/bin/env python

import os
import pathlib
import requests
import tempfile
import shutil

requests.packages.urllib3.disable_warnings()

root_dir = pathlib.Path(os.path.realpath(__file__)).parent
data_dir = root_dir / 'data'

def load_data(url, datasets, **kwargs):
    archive_suffix = ''.join(pathlib.Path(url).suffixes)
    with tempfile.NamedTemporaryFile('wb', suffix=archive_suffix) as archive_file:
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
                dst.mkdir(parents=True, exists_ok=True)
                shutil.move(str(src_root / src), str(data_dir / dst))

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

load_data(
    'https://bmcv.github.io/SuperDSM/fibroblast-prolif.tar.bz2',
    [
        ('fibroblast-prolif/prolif', 'fibroblast/prolif')
    ]
)

load_data(
    'https://bmcv.github.io/SuperDSM/fibroblast-ss.tar.bz2',
    [
        ('fibroblast-ss/ss', 'fibroblast/ss')
    ]
)

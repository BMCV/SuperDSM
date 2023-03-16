import os
import pathlib
import requests
import tempfile
import shutil
import json
import warnings
import numpy as np
import superdsm.io

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
                dst = data_dir / dst
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_root / src), str(dst))

def require_data(data_id, filename=None):
    data_path = data_dir / data_id
    if not data_path.exists():
        with open(data_dir.parent / 'database.json') as dbfp:
            db = json.load(dbfp)
        data = db[data_id]
        load_data(data['url'], [(src, f'{data_id}/{dst}') for (src, dst) in data['datasets']])
    return data_path if filename is None else data_path / filename


def normalize_image(img):
    if str(img.dtype).startswith('float'):
        img = (img - img.min()) / (img.max() - img.min()) 
        img = (img * 255).round().astype('uint8')
    return img


def validate_image(test, name, img):
    try:
        expected = superdsm.io.imread(str(root_dir / 'expected' / name), as_gray=False)
        img = normalize_image(img) ## processes the image as if it was written to a file and read back
        test.assertEqual(img.shape, expected.shape)
        msg = f'Maximum absolute difference: {np.abs(img - expected).max():g}'
        test.assertTrue(np.allclose(img, expected), msg)
    except:
        actual_path = root_dir / 'actual' / name
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        superdsm.io.imwrite(str(actual_path), img)
        raise


def without_resource_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test
    

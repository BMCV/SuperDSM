import sys
import argparse
import glob
import pathlib
import skimage.io
import csv
import scipy.ndimage as ndi
import shutil


parser = argparse.ArgumentParser(prog='Regression testing')

parser.add_argument(  'actual_seg', help='Directory containing the actual label maps.')
parser.add_argument(  'actual_csv', help='Directory where the actual CSV should be written.')
parser.add_argument('expected_csv', help='Directory containing the expected CSV.')
parser.add_argument('--update-expected', help='Update the expected CSV.', action='store_true')

args = parser.parse_args()


actual_seg_path   = pathlib.Path(args.actual_seg)
actual_csv_path   = pathlib.Path(args.actual_csv)
expected_csv_path = pathlib.Path(args.expected_csv)


actual_csv_rows_by_filename = dict()
for filepath in glob.glob(str(actual_seg_path / '*.png')):
    actual_seg_filepath = pathlib.Path(filepath)
    actual_csv_filepath = actual_csv_path / (actual_seg_filepath.name + '.csv')

    img = skimage.io.imread(str(actual_seg_filepath))
    actual_csv_rows = list()
    for l in frozenset(img.flatten()) - {0}:
        cc = (img == l)
        cc_center_rc = ndi.center_of_mass(cc)
        actual_csv_rows.append((str(cc.sum()), str(round(cc_center_rc[1], 1)), str(round(cc_center_rc[0], 1))))

    actual_csv_rows.sort(key = lambda row: row[1:3])
    with actual_csv_filepath.open('w') as fp:
        writer = csv.writer(fp, delimiter = ',', quoting = csv.QUOTE_ALL)
        writer.writerows([['Object size', 'Center X', 'Center Y']] + actual_csv_rows)

    actual_csv_rows_by_filename[actual_seg_filepath.name] = frozenset(actual_csv_rows)
    sys.stdout.write('.')
    sys.stdout.flush()

sys.stdout.write('\n')


if args.update_expected:
    for filename in actual_csv_rows_by_filename.keys():
        csv_filename = filename + '.csv'
        shutil.move(str(actual_csv_path / csv_filename), str(expected_csv_path / csv_filename))


errors = list()
for filepath in glob.glob(str(expected_csv_path / '*.csv')):
    expected_csv_filepath = pathlib.Path(filepath)
    actual_seg_filename = expected_csv_filepath.name[:-4]

    try:
        actual_csv_rows = actual_csv_rows_by_filename.pop(actual_seg_filename)
    except KeyError:
        errors.append(f'Missing label map: "{actual_seg_filename}"')
        continue

    expected_csv_rows = list()
    with expected_csv_filepath.open('r') as fp:
        reader = csv.reader(fp, delimiter = ',', quoting = csv.QUOTE_ALL)
        for ridx, row in enumerate(reader):
            if ridx == 0: continue  ## skip header
            expected_csv_rows.append(tuple(row))

    expected_csv_rows = frozenset(expected_csv_rows)
    missing_rows  = expected_csv_rows - actual_csv_rows
    spurious_rows = actual_csv_rows - expected_csv_rows

    if len(spurious_rows) > 0 or len(missing_rows) > 0:
        errors.append(f'{actual_seg_filename}: {len(spurious_rows)} spurious object(s) and {len(missing_rows)} missing object(s) cannot be matched')


for remaining_filename in actual_csv_rows_by_filename.keys():
    errors.append(f'Spurious label map: "{remaining_filename}"')


if 'img' not in locals():  ## an additional error check to prevent passing tests due to wrongs paths
    errors.append('No label maps found')

if len(errors) == 0:
    print(f'All tests passed.')
    print()
    sys.exit(0)

else:
    print(f'{len(errors)} test(s) failed:')
    for error in errors:
        print(f'- {error}')
    print()
    sys.exit(1)

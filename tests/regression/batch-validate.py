import argparse
import os
import pathlib
import socket


parser = argparse.ArgumentParser(prog='Regression testing (batch)')

parser.add_argument('actual_csv', help='Root directory where the actual CSV will be stored.')
parser.add_argument('--taskdirs', help='Directory for which the /seg subdirectory will be validated.', nargs='+', default=list())
parser.add_argument('--update-expected', help='Update the expected CSV.', action='store_true')

args = parser.parse_args()

hostname = socket.gethostname()

options = '--update-expected' if args.update_expected else ''

for taskdir in args.taskdirs:

    print(f'\nValidating: {taskdir}')

    actual_csv = pathlib.Path(args.actual_csv) / taskdir
    actual_csv.mkdir(parents=True, exist_ok=True)
    os.system(f'python tests/regression/validate.py "examples/{taskdir}/seg" "{str(actual_csv)}" "tests/regression/expected/{hostname}/{taskdir}" {options}')

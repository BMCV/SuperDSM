#!/bin/bash

set -e

python tests/regression/batch-validate.py "$@" --taskdirs "BBBC039" "BBBC039/isbi24"

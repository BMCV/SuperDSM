#!/bin/bash

set -e

python tests/regression/batch-validate.py "$@" --taskdirs "U2OS/default" "U2OS/default/adapted" "U2OS/default/adapted/isbi24"

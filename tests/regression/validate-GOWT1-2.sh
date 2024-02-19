#!/bin/bash

set -e

python tests/regression/batch-validate.py "$@" --taskdirs "GOWT1-2/default" "GOWT1-2/default/adapted" "GOWT1-2/default/adapted/isbi24"

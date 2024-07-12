#!/bin/bash

set -e

python tests/regression/batch-validate.py "$@" --taskdirs "GOWT1-1/default" "GOWT1-1/default/adapted" "GOWT1-1/default/adapted/isbi24"

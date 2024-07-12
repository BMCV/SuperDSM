#!/bin/bash

set -e

python tests/regression/batch-validate.py "$@" --taskdirs "NIH3T3/default" "NIH3T3/default/adapted" "NIH3T3/default/adapted/isbi24"

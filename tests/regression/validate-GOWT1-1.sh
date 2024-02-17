#!/bin/bash

set -e

for taskdir in "GOWT1-1/default" "GOWT1-1/default/adapted" "GOWT1-1/default/adapted/isbi24"
do

    mkdir -p "$1/${taskdir}"
    python tests/regression/validate.py \
      "examples/${taskdir}/seg" \
      "$1/${taskdir}" \
      "tests/regression/expected/$(hostname)/${taskdir}"

done

#!/bin/bash

set -e

for taskdir in "U2OS/default" "U2OS/default/adapted" "U2OS/default/adapted/isbi24"
do

    mkdir -p "$1/${taskdir}"
    python tests/regression/validate.py \
      "examples/${taskdir}/seg" \
      "$1/${taskdir}" \
      "tests/regression/expected/$(hostname)/${taskdir}"

done

#!/bin/bash

set -e

for taskdir in "NIH3T3/default" "NIH3T3/default/adapted" "NIH3T3/default/adapted/isbi24"
do

    mkdir -p "$1/$(taskdir)"
    python tests/regression/validate.py \
      "examples/$(taskdir)/seg" \
      "$1/$(taskdir)" \
      "tests/regression/expected/$(hostname)/$(taskdir)"

done

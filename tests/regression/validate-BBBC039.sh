#!/bin/bash

set -e

for taskdir in "BBBC039" "BBBC039/isbi24"
do

    mkdir -p "$1/$(taskdir)"
    python tests/regression/validate.py \
      "examples/$(taskdir)/seg" \
      "$1/$(taskdir)" \
      "tests/regression/expected/$(hostname)/$(taskdir)"

done

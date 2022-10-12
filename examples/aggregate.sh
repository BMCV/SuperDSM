#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate kostrykin2021-amd
./aggregate.py

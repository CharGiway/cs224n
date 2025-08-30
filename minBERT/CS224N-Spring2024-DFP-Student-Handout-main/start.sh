#!/usr/bin/env bash
set -e

python3 classifier.py --fine-tune-mode full-model --lr 1e-5

# python3 classifier.py --fine-tune-mode last-linear-layer --lr 1e-3

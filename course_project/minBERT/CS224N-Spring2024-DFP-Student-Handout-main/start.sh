#!/usr/bin/env bash
set -e
export HF_ENDPOINT="https://hf-mirror.com"

# python3 classifier.py --fine-tune-mode full-model --lr 1e-5 --use_gpu

python3 classifier.py --fine-tune-mode last-linear-layer --lr 1e-3 --use_gpu

#!/usr/bin/env bash

srun -p pat_earth --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=eval --kill-on-bad-exit=1
python eval.py  --ngpu=2 --dataset=h36m-p2 --batch_size=256 --checkpoint= --config=
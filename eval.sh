#!/usr/bin/env bash

srun -p pat_earth --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=eval --kill-on-bad-exit=1
python eval.py  --ngpu=2 --dataset=h36m-p2 --batch_size=256 --checkpoint= --config=


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth eval 8 ./tools/pose.sh
--eval --val_dataset=h36m-p2 --batch_size=128 --num_workers=4 --name=eval
--model=hmr --resume_from=logs/hmr/checkpoints/checkpoint0029.pth
--model=pvt_small --resume_from=logs/pvts_all1/checkpoint0199.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth eval 8 ./tools/pose.sh
--eval --val_dataset=h36m-p2 --batch_size=128 --num_workers=4 --name=eval
--model=pvt_small --resume_from=logs/pvts_all1/checkpoints/checkpoint0199.pth

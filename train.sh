#!/usr/bin/env bash

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=200 --checkpoint_steps=10000
--name=pvt_medium --model=pvt_medium

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=50 --checkpoint_steps=10000
--name=mypvt_medium --model=mypvt_medium

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_medium 8 ./tools/pose.sh
--dataset=mesh --batch_size=32 --num_workers=2 --num_epochs=300 --summary_steps=100
--name=pvt_medium2 --model=pvt_medium

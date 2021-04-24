#!/usr/bin/env bash

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=200 --checkpoint_steps=10000
--name=pvt_medium --model=pvt_medium

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=50 --checkpoint_steps=10000
--name=mypvt_medium --model=mypvt_medium


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_m 8 ./tools/pose.sh
--dataset=mesh --batch_size=64 --num_workers=4 --num_epochs=50 --summary_steps=100
--name=pvt_medium3 --model=pvt_medium --opt=adamw --lr=1e-4 --wd=0.05










GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt 8 ./tools/pose.sh
--dataset=mesh --batch_size=64 --num_workers=4 --num_epochs=30 --summary_steps=50
--name=hmr --model=hmr

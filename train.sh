#!/usr/bin/env bash

python train.py --name=tran1 --dataset=mesh --batch_size=128 --ngpu=8 --num_workers=16 --num_epochs=50 --enc_layers=0

python train.py --name=tran2 --dataset=mesh --batch_size=128 --ngpu=8 --num_workers=16 --num_epochs=50 --enc_layers=2


python train.py --name=tran3 --dataset=mesh --batch_size=128 --ngpu=8 --num_workers=16 --num_epochs=50 --enc_layers=0
--checkpoint_steps=20000 --use_inter --pose_head=share


python train.py --name=tran4 --dataset=mesh --batch_size=128 --ngpu=8 --num_workers=16 --num_epochs=50 --enc_layers=0
--checkpoint_steps=20000 --use_inter --pose_head=specific


srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=eval --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=200 --checkpoint_steps=10000
--name=pvt_medium --model=pvt_medium

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=eval --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=50 --checkpoint_steps=10000
--name=mypvt_medium --model=mypvt_medium

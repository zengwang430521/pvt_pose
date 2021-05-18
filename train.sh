#!/usr/bin/env bash


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt 8 ./tools/pose.sh
--dataset=mesh --batch_size=64 --num_workers=4 --num_epochs=40 --summary_steps=50
--name=hmr --model=hmr --resume_from=logs/hmr/checkpoints/checkpoint0029.pth


srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=200 --checkpoint_steps=10000
--name=pvt_medium --model=pvt_medium

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_pose --kill-on-bad-exit=1
python train.py --dataset=mesh --batch_size=256 --ngpu=8 --num_workers=16 --num_epochs=50 --checkpoint_steps=10000
--name=mypvt_medium --model=mypvt_medium


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_m 8 ./tools/pose.sh
--dataset=mesh --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100
--name=pvt_medium4 --model=pvt_medium --opt=adamw --lr=1e-4 --wd=0.05
--pretrain_from=data/pretrained/pvt_medium.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_m 8 ./tools/pose.sh
--dataset=mesh --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100
--name=pvt_medium5 --model=pvt_medium --opt=adamw --lr=5e-4  --lr_drop=50 --wd=0.05
--pretrain_from=data/pretrained/pvt_medium.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvt_m_all --model=pvt_medium --opt=adamw --lr=2.5e-4 --wd=0.05
--pretrain_from=data/pretrained/pvt_medium.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100
--name=pvt_m_all_2 --model=pvt_medium --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80
--resume_from=logs/pvt_m_all/checkpoints/checkpoint0079.pth



GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvts_all1 --model=pvt_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=150
--pretrain_from=data/pretrained/pvt_small.pth

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100
--name=pvts_all2 --model=pvt_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=150
--pvt_alpha=0.3
--pretrain_from=data/pretrained/pvt_small.pth
--resume_from=logs/pvts_all2/checkpoints/checkpoint0099.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvtsnc_all1 --model=pvt_nc_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=150
--pretrain_from=data/pretrained/pvt_small.pth --pvt_alpha=1



GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvtsnc2_all1 --model=pvt_nc2_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=150
--pretrain_from=data/pretrained/pvt_small.pth --pvt_alpha=1


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=150 --summary_steps=100
--name=pvt2048_small --model=pvt2048_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--pretrain_from=data/pretrained/pvt_small.pth --pvt_alpha=1


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvts_all3 --model=pvt_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 --backbone_lr=0.1
--pretrain_from=data/pretrained/pvt_small.pth

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvts_all4 --model=pvt_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 --backbone_lr=0.1
--pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/pvt_s/checkpoint.pth



GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=pvts_all5 --model=pvt_small --opt=adamw --lr=1e-3 --wd=0.05 --lr_drop=100 --backbone_lr=0.1
--pretrain_from=data/pretrained/pvt_small.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=mypvt2_all1 --model=mypvt2_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 --backbone_lr=0.1
--pretrain_from=data/pretrained/mypvt2_small_108.pth



GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=200 --summary_steps=100
--name=mypvt2_all2 --model=mypvt2_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--pretrain_from=data/pretrained/mypvt2_small_108.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=150 --summary_steps=100
--name=pvtimps_all1 --model=pvt_small_impr1_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--pretrain_from=data/pretrained/pvt_small_impr1_peg.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 16 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=150 --summary_steps=100
--name=pvtimps_all2 --model=pvt_small_impr1_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--pretrain_from=data/pretrained/pvt_small_impr1_peg.pth
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


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=150 --summary_steps=100
--name=my9_all --model=mypvt9_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--resume_from=logs/my9_all/checkpoints/checkpoint0009.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=150 --summary_steps=100
--name=my14_3_all_300 --model=mypvt14_3_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100
--pretrain_from=data/pretrained/mypvt14_3_300.pth

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth pvt_pose 8 ./tools/pose.sh
--dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100
--name=p2_all_300 --model=pvt_small_impr8_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
--pretrain_from=data/pretrained/pvt_small_impr8_peg.pth \
--resume_from=logs/p2_all_300/checkpoint/checkpoint_latest.pth


    -x SH-IDC1-10-198-4-100,SH-IDC1-10-198-4-101,SH-IDC1-10-198-4-102,SH-IDC1-10-198-4-103,SH-IDC1-10-198-4-116,SH-IDC1-10-198-4-117,SH-IDC1-10-198-4-118,SH-IDC1-10-198-4-119 \

srun -p pat_earth \
    --ntasks 8 \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=p2_all_300 --model=pvt_small_impr8_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/pvt_small_impr8_peg.pth \
    --resume_from=logs/p2_all_300/checkpoint/checkpoint_latest.pth


srun -p pat_earth \
    --ntasks 8 \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_all --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_all/checkpoints/checkpoint_latest.pth     --img_res=448


srun -p 3dv-share -w SH-IDC1-10-198-6-130\
    --ntasks 4 \
    --job-name=debug \
    --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=1 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all \
    --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=mypvt2520_2_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --resume_from=logs/my2520_all/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc --eval


    python -u main.py --dataset=spin --use_spin_fit --adaptive_weight --gtkey3d_from_mesh \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_spin --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_spin/checkpoints/checkpoint_latest.pth     --img_res=448


srun -p 3dv-share -w SH-IDC1-10-198-6-129 \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks 8 \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=4 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20f_all --model=mypvt20_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20f_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my20_f/checkpoint.pth

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_all --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_300.pth



    python -u main.py --dataset=spin --use_spin_fit --adaptive_weight --gtkey3d_from_mesh \
    --batch_size=32 --num_workers=4 --num_epochs=110 --summary_steps=100 \
    --name=my2520_spin --model=mypvt2520_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/my2520_300.pth \
    --resume_from=logs/my2520_spin/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc

    python -u main.py --dataset=spin
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_spin --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_spin/checkpoints/checkpoint_latest.pth     --img_res=448

    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_3_all --model=mypvt2520_3_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=90 \
    --pretrain_from=data/pretrained/my2520_3_320.pth \
    --resume_from=logs/my2520_3_all/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc

    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520g_all --model=mypvt2520g_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=90 \
    --pretrain_from=data/pretrained/my2520g_318.pth \
    --resume_from=logs/my2520g_all/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc


    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_2_all --model=mypvt2520_2_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2520_300.pth \
    --resume_from=logs/my2520_2_all/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=110 --summary_steps=100 \
    --name=my2520_spin --model=mypvt2520_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/my2520_300.pth \
    --resume_from=logs/my2520_spin/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc


    python -u main_finetune.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=110 --summary_steps=100 \
    --name=my20_all_5 --model=mypvt20_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=100 \
    --resume_from=logs/my20_all_4/checkpoints/checkpoint0079.pth     --img_res=224 --use_mc \

    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_all --model=mypvt2520_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2520_300.pth \
    --resume_from=logs/my2520_all/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_all_4 --model=mypvt20_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=80 \
    --resume_from=logs/my20_all_4/checkpoints/checkpoint_latest.pth     --img_res=224 \




    python -u main.py --dataset=spin --use_spin_fit --adaptive_weight --gtkey3d_from_mesh \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_spin --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_spin/checkpoints/checkpoint_latest.pth     --img_res=448


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_all_2 --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_all_2/checkpoints/checkpoint_latest.pth     --img_res=448


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_all --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=80 \
    --resume_from=logs/my2320_all/checkpoints/checkpoint_0079.pth     --img_res=448





    python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=pvt_small_impr8_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/pvt_small_impr8_peg.pth

    python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_all --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_300.pth \
    --resume_from=logs/my2320_all/checkpoints/checkpoint_latest.pth     --img_res=448




spring.submit arun \
    -p spring_scheduler \
    -n8 --gpu \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    "python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2320_all_300 --model=mypvt2320_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=80 \
    --pretrain_from=data/pretrained/my2320_287.pth \
    --resume_from=logs/my2320_all_300/checkpoints/checkpoint_latest.pth \
    --img_res=448"

    "python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=p2_all_300 --model=pvt_small_impr8_peg --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/pvt_small_impr8_peg.pth \
    --resume_from=logs/p2_all_300/checkpoints/checkpoint_latest.pth"


srun -p pat_earth \
    -x SH-IDC1-10-198-4-100,SH-IDC1-10-198-4-101,SH-IDC1-10-198-4-102,SH-IDC1-10-198-4-103,SH-IDC1-10-198-4-116,SH-IDC1-10-198-4-117,SH-IDC1-10-198-4-118,SH-IDC1-10-198-4-119 \
    --ntasks 8 \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my21_fine/checkpoint.pth \
    --name=my21_all_f350 --model=mypvt21_small --opt=adamw --lr=2.5e-4 --wd=0.0001 --lr_drop=100 \
    --resume_from=logs/my21_all_f350/checkpoints/checkpoint_latest.pth \
    --img_res=448

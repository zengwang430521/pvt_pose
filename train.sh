#!/usr/bin/env bash
spring.submit arun -p spring_scheduler --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    -n1 --gpu --job-name=mesh --gpu-type 16gv100  -w SH-IDC1-10-198-6-245 \
    " ls "

spring.submit arun -p spring_scheduler --gres=gpu:7 --ntasks-per-node=7 --cpus-per-task=5 \
    -n7 --gpu --job-name=mesh --gpu-type 16gv100  -w SH-IDC1-10-198-6-245 \
    "
    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=60 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=30  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt_again  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0_mix1_hiratt_again/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002     --eval_freq=2 --loss_type=4
    "


export MASTER_PORT=29501

    --ntasks 4 --job-name=mesh --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --ntasks 1 --job-name=mesh --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --ntasks 8 --job-name=mesh --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \




srun -p 3dv-share -w SH-IDC1-10-198-6-129\
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto\
srun -p pat_earth \
    --ntasks 8 --job-name=mesh --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    \
    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=60 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=50  --img_res=224 --use_mc \
    --model=tcformer_small --dataset=mix1 --head_type=hiratt_hmr3 \
    --name=tc_mix1_att3_l4  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --loss_type=4 --resume_from=logs/tc_mix1_att3_l4/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002     --eval_freq=1  --val_dataset=3dpw

    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=60 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=50  --img_res=224 --use_mc \
    --model=tcformer_small --dataset=mix1 --head_type=mta_hmr \
    --name=tc_mix1_mta_l4  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --loss_type=4 --resume_from=logs/tc_mix1_mta_l4/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002     --eval_freq=1  --val_dataset=3dpw


    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hmr --dataset=mix1 \
     --name=tcformer_wo_att_loss7_fine --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_wo_att_loss5/checkpoints/checkpoint_best.pth \
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=7


    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hmr --dataset=mix1 \
     --name=tcformer_wo_att_loss5_fine --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_wo_att/checkpoints/checkpoint_best.pth \
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=5


    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hmr --dataset=mix1 \
     --name=tcformer_wo_att_loss5 --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_wo_att_loss5/checkpoints/checkpoint_latest.pth \
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=5


    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hiratt_hmr --dataset=mix1\
    --name=tcformer_loss5 --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_loss5/checkpoints/checkpoint_latest.pth\
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=5


     python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=6 \
    --model=tcformer_small  --head_type=hmr --dataset=mix1\
    --name=tcformer_wo_att_loss6 --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_wo_att_loss6/checkpoints/checkpoint_latest.pth\




    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hiratt_hmr2 --dataset=mix1\
    --name=tcformer_att2 --log_dir='./logs/rebuttal' \
    --resume_from=logs/rebuttal/tcformer_att2/checkpoints/checkpoint_latest.pth\
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=4






    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_grid_small  --head_type=hiratt_hmr --dataset=mix1\
     --name=tcformer_wo_cluster --log_dir='./logs/rebuttal' \
     --resume_from=logs/rebuttal/tcformer/checkpoints/checkpoint_latest.pth\
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=4


    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=tcformer_small  --head_type=hmr --dataset=mix1\
     --name=tcformer_wo_att --log_dir='./logs/rebuttal' \
     --resume_from=logs/rebuttal/tcformer_wo_att/checkpoints/checkpoint_latest.pth\
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=4




    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small  --head_type=hiratt_hmr \
    --dataset=mix1 --name=mix1_1  --resume_from=logs/mix1_1/checkpoints/checkpoint_latest.pth\
    --val_dataset=3dpw  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=4



    python -u main.py --batch_size=64 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=70  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix2 --head_type=hiratt_hmr \
    --name=mix2_1  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/mix2_1/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002  --eval_freq=1 --loss_type=4


    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adam --lr=1e-4 --lr_drop=90  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt_fine3  \
    --val_dataset=3dpw --pretrain_from=logs/den0_mix1_hiratt/checkpoints/checkpoint0069.pth \
    --lam_smpl_beta=0.002     --eval_freq=1 --loss_type=4 --lam_mesh=1 --lam_key3d=2 --use_smpl_joints --lam_key2d_smpl=0 --lam_key3d_smpl=1


    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adam --lr=1e-4 --lr_drop=90  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt_fine2  \
    --val_dataset=3dpw --resume_from=logs/den0_mix1_hiratt/checkpoints/checkpoint0069.pth \
    --lam_smpl_beta=0.002     --eval_freq=1 --loss_type=4 --lam_mesh=0.5 --lam_key3d=2

    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=90  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt_fine  \
    --val_dataset=3dpw --resume_from=logs/den0_mix1_hiratt/checkpoints/checkpoint0069.pth \
    --lam_smpl_beta=0.002     --eval_freq=1 --loss_type=4


    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=60 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=30  --img_res=224 --use_mc \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt_again  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0_mix1_hiratt_again/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002     --eval_freq=1 --loss_type=4



    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=70  --lr_drop=50 --summary_steps=100 \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --opt=adamw --lr=5e-5 --wd=0.01   --img_res=224 --use_mc \
    --lam_smpl_beta=0.06 --lam_smpl_pose=60 --lam_mesh=0 --lam_key2d=300 --lam_key3d=300  \
    --name=den0f_mix1_hiratt  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0f_mix1_hiratt/checkpoints/checkpoint_latest.pth \
    --eval_freq=2





    python -u main.py --batch_size=32 --num_workers=5 --num_epochs=100 --summary_steps=100 \
    --opt=adamw --lr=1e-4 --wd=0.01 --lr_drop=90  --img_res=224 --use_mc \
     \
    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=den0_mix1_hiratt  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0_mix1_hiratt/checkpoints/checkpoint_latest.pth \
    --lam_smpl_beta=0.002


    --model=mypvt3h2_densityc_small --dataset=mix1 --head_type=hiratt_hmr \
    --name=denc_mix1_hiratt  --pretrain_from=data/pretrained/3h2_densityc_small.pth\
    --val_dataset=3dpw --resume_from=logs/denc_mix1_hiratt/checkpoints/checkpoint_latest.pth

     --model=mypvt3h2_density0_small --dataset=mix1 --head_type=att_hmr \



    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=att_hmr \
    --name=den0_mix1_att  --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0_mix1_att/checkpoints/checkpoint_latest.pth

    --model=mypvt3h2_density0_small --dataset=mix1 --head_type=hmr \
    --name=den0_mix1 --model=mypvt3h2_density0_small --pretrain_from=data/pretrained/3h2_density0_small.pth\
    --val_dataset=3dpw --resume_from=logs/den0_mix1/checkpoints/checkpoint_latest.pth


































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



srun -p 3dv-share \
    --ntasks 1 --job-name=debug \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 \

spring.submit arun -p spring_scheduler -n1 --gpu \
    --job-name=debug \
    --gres=gpu:2 --ntasks-per-node=1 --cpus-per-task=1 \
    'python -m pytest test.py'




srun -p 3dv-share\
    --ntasks 1 --job-name=spin \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --kill-on-bad-exit=1 \
     python -u main.py --dataset=h36m \
     --batch_size=64 --num_workers=4 --model=hmr --opt=adamw \
     --name=debug --run_smplify --iter_smplify=0 \
     --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc --eval\
     --resume_from=logs/hmr_opt_f/checkpoints/checkpoint_latest.pth


export MASTER_PORT=29505
srun -p 3dv-share -w SH-IDC1-10-198-6-130 \
    --ntasks 1 --job-name=spin \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --kill-on-bad-exit=1 \
     python -u main.py --dataset=h36m \
     --batch_size=64 --num_workers=4 --model=hmr --opt=adamw \
     --name=debug --run_smplify --iter_smplify=0 \
     --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc --eval --val_dataset=3dpw\
     --resume_from=logs/hmr_opt_f2/checkpoints/checkpoint_latest.pth \


    python -u main.py --dataset=all --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2c_all_f --model=mypvt20_2c_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2c_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc --eval

      python -u main.py --dataset=spin \
     --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
     --model=hmr --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
      --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
     --name=hmr_opt_f2 --run_smplify --iter_smplify=0 \
     --resume_from=logs/hmr_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
     --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc

    python -u main.py --dataset=spin \
    --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_opt_f2 --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
    --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/my20_2f_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --use_mc

     python -u main.py --dataset=spin \
     --batch_size=64 --num_workers=4 --num_epochs=100 --summary_steps=100 \
     --name=hmr_opt_f --run_smplify --iter_smplify=50 \
     --model=hmr --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
      --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
     --resume_from=logs/hmr_opt_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
     --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc




srun -p 3dv-share --ntasks 1 --job-name=make --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 sh make.sh

spring.submit arun \
    -p spring_scheduler \
    -n8 --gpu \
    --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \

spring.submit arun\
    -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    -n8 --gpu --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=4 \

srun -p pat_earth \
srun -p 3dv-share -x SH-IDC1-10-198-6-[132-135] \
srun -p 3dv-share -w SH-IDC1-10-198-6-129\
srun -p pat_earth \
    -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks 8 --job-name=mesh \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=4 --kill-on-bad-exit=1 \
    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_eft --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_eft/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_eft_f --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_eft_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc --eval

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2d_all_f --model=mypvt20_2d_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2d_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc --eval

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=60 --summary_steps=100 \
    --name=my20_8_all_f --model=mypvt20_8_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=50 \
    --resume_from=logs/my20_8_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2a_all_f --model=mypvt20_2a_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2a_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

     python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_9_all_f --model=mypvt20_9_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_9_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_7a_all_f --model=mypvt20_7a_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_7a_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=24 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_7_all_f --model=mypvt20_7_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_7_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4f_all_f --model=mypvt20_4f_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4f_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4d3_all_f --model=mypvt20_4d3_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4d3_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4e_all_f --model=mypvt20_4e_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4e_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=24 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_all_f --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4d2_all_f --model=mypvt20_4d2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4d2_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4e_all_f --model=mypvt20_4e_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4e_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=24 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_6_all_f --model=mypvt20_6_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_6_all_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc




    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_3a_all_f --model=mypvt20_3a_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_3a_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4d_all_f --model=mypvt20_4d_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4d_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_opt_f2 --run_smplify --iter_smplify=50 \
    --model=hmr --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
     --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/hmr_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc

    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_opt_f2 --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
    --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/my20_2f_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4d_all_f --model=mypvt20_4d_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4d_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=spin \
    --batch_size=8 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_opt_f --run_smplify --iter_smplify=50 \
    --model=hmr --opt=adamw --lr=3e-5 --wd=1e-4 --lr_drop=90 \
     --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/hmr_opt_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4c_all_f --model=mypvt20_4c_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4c_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4_all_f --model=mypvt20_4_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc



   python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_opt_f2 --run_smplify --iter_smplify=50 \
    --model=hmr --opt=adamw --lr=12e-5 --wd=1e-4 --lr_drop=90 \
     --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/hmr_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/hmr_all/checkpoints/checkpoint_latest.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4c_all_f --model=mypvt20_4c_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4c_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_4b_all_f --model=mypvt20_4b_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_4b_all_f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_opt_f2 --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=1.2e-4 --wd=1e-4 --lr_drop=90 \
    --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/my20_2f_opt_f2/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_5_all --model=mypvt20_5_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_5_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=10 \
    --name=my20_2f_opt_f --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_opt_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=all --batch_size=12 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_all_448  --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_all_448/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_3_all  --model=mypvt20_3_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_3_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2b_all_f  --model=mypvt20_2b_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2b_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc --eval

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_3_all_f  --model=mypvt20_3_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_3_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc



    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_opt --run_smplify --iter_smplify=100 \
    --model=hmr --opt=adamw --lr=5e-5 --wd=1e-4 --lr_drop=90 \
    --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --resume_from=logs/hmr_opt/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --use_mc


    python -u main.py --dataset=all --batch_size=12 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_all_448f  --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_all_448f/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_debug --run_smplify --iter_smplify=1 \
    --model=hmr --opt=adamw --lr=5e-5 --wd=1e-4 --lr_drop=90 \
    --lam_mesh=0 --lam_key2d=300 --lam_key3d=300 --lam_smpl_pose=60 --lam_smpl_beta=0.060 --lam_camera=60 \
    --img_res=224 --use_mc

    python -u main.py --dataset=all --batch_size=12 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_all_448  --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_all_448/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth --use_mc


    python -u main.py --dataset=all --batch_size=16 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_all  --model=hmr --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/hmr_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_11_all --model=mypvt2520_11_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_11_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=10 \
    --name=my20_2f_opt --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_opt/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc

    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=hmr \
    --resume_from=logs/hmr/checkpoints/checkpoint0049.pth \
    --img_res=448  --eval --use_mc

    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=mypvt20_2_small \
    --resume_from=logs/my20_2f_all2/checkpoints/checkpoint0099.pth \
    --img_res=224  --eval --use_mc


    python -u main.py --dataset=all --batch_size=16 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=hmr_all_448  --model=hmr --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/hmr_all_448/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=10 \
    --name=my20_2f_opt --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_opt/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=all --batch_size=16 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_all_448  --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_all_448/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_5f_all --model=mypvt2520_5_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_5f_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my2520_5_330.pth --use_mc


    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_opt --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_opt/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=all \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=debug --model=mypvt2520_7_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_7_f_all/checkpoints/checkpoint_latest.pth  --img_res=896  --eval --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_10_all --model=mypvt2520_10_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_10_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=spin --use_spin_fit --adaptive_weight --gtkey3d_from_mesh \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_spin --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_spin/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_7_f_all --model=mypvt2520_7_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_7_f_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my2520_7_f/checkpoint.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_8_f_all --model=mypvt2520_8_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_8_f_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my2520_8_f/checkpoint.pth --use_mc


    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_9_f_all --model=mypvt2520_9_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_9_f_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my2520_9_f/checkpoint.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_9_all --model=mypvt2520_9_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_9_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_8_all --model=mypvt2520_8_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_8_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_7_all --model=mypvt2520_7_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_7_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my2520_4_all --model=mypvt2520_4_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my2520_4_all/checkpoints/checkpoint_latest.pth     --img_res=448 \
    --pretrain_from=data/pretrained/my20_300_pre.pth --use_mc

    python -u main_finetune.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2f_all2 --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_all/checkpoints/checkpoint0049.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc




    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=60 --summary_steps=100 \
    --name=my20_2_all2 --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=50 \
    --resume_from=logs/my20_2_all/checkpoints/checkpoint0049.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_300.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=60 --summary_steps=100 \
    --name=my20_2f_all --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=50 \
    --resume_from=logs/my20_2f_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_2_330.pth --use_mc

    python -u main.py --dataset=spin --use_spin_fit --adaptive_weight --gtkey3d_from_mesh \
    --batch_size=32 --num_workers=4 --num_epochs=110 --summary_steps=100 \
    --name=my2520_spin --model=mypvt2520_small --opt=adamw --lr=2.5e-4 --wd=0.05 --lr_drop=100 \
    --pretrain_from=data/pretrained/my2520_300.pth \
    --resume_from=logs/my2520_spin/checkpoints/checkpoint_latest.pth     --img_res=448 --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_all_tcmr --model=mypvt20_2_small --head_type=tcmr --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_all_tcmr/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_300.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_all_cmr --model=mypvt20_small --head_type=cmr --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_all_cmr/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_300.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20_2_all --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=data/pretrained/my20_300.pth --use_mc

    python -u main.py --dataset=all --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=100 \
    --name=my20f_all --model=mypvt20_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20f_all/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --pretrain_from=/mnt/lustre/zengwang/codes/PVT/work_dirs/my20_f/checkpoint.pth





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

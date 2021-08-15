#!/usr/bin/env bash


srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks 1 --job-name=mesh \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --kill-on-bad-exit=1 \
    python -u main.py --dataset=spin \
    --batch_size=32 --num_workers=4 --num_epochs=100 --summary_steps=10 \
    --name=my20_2f_opt_f --run_smplify --iter_smplify=50 \
    --model=mypvt20_2_small --opt=adamw --lr=2.5e-4 --wd=1e-4 --lr_drop=90 \
    --resume_from=logs/my20_2f_opt_f/checkpoints/checkpoint_latest.pth     --img_res=224 \
    --eval --use_mc



srun -p pat_earth --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=eval --kill-on-bad-exit=1
python eval.py  --ngpu=2 --dataset=h36m-p2 --batch_size=256 --checkpoint= --config=


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth eval 8 ./tools/pose.sh
--eval --val_dataset=h36m-p2 --batch_size=128 --num_workers=4 --name=eval
--model=hmr --resume_from=logs/hmr/checkpoints/checkpoint0029.pth
--model=pvt_small --resume_from=logs/pvts_all1/checkpoint0199.pth


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh pat_earth eval 8 ./tools/pose.sh
--eval --val_dataset=h36m-p2 --batch_size=128 --num_workers=4 --name=eval
--model=pvt_small --resume_from=logs/pvts_all1/checkpoints/checkpoint0199.pth




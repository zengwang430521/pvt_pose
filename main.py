import warnings
warnings.simplefilter("ignore", UserWarning)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import datetime
import json
import random
import time
import numpy as np
import torch
import utils.misc as utils
import utils.samplers as samplers
from torch.utils.data import DataLoader
from train.train_one_epoch import train_one_epoch, evaluate
from pathlib import Path
from train.criterion import MeshLoss, JointEvaluator
from models.TMR import build_model
from datasets.datasets import create_dataset, create_val_dataset
from utils.train_options import DDPTrainOptions
from tensorboardX import SummaryWriter
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer


def build_optimizer(model, options):
    if options.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            params=list(model.parameters()),
            lr=options.lr,
            betas=(options.adam_beta1, 0.999),
            weight_decay=options.wd)
    else:
        optimizer = torch.optim.Adam(
            params=list(model.parameters()),
            lr=options.lr,
            betas=(options.adam_beta1, 0.999),
            weight_decay=options.wd)
    return optimizer


def build_scheduler(optimizer, options):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.lr_drop)
    return lr_scheduler


def main(options):
    utils.init_distributed_mode(options)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(options)

    # summary writer
    if utils.is_main_process() and (not options.eval):
        summary_writer = SummaryWriter(options.summary_dir)
        summary_writer.iter_num = 0
        print('summary writer created')
    else:
        summary_writer = None

    # device = torch.device('cuda')
    device = torch.device(options.device)

    # fix the seed for reproducibility
    seed = options.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(options)
    model = build_model(options)
    model.to(device)
    criterion = MeshLoss(options, device)
    evaluator = JointEvaluator(options, device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    print('start build dataset')
    dataset_train = create_dataset(options.dataset, options)
    dataset_val = create_val_dataset(options.val_dataset, options)
    print('finish build dataset')

    if options.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, options.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=options.num_workers,
                                   pin_memory=True)

    data_loader_val = DataLoader(dataset_val, options.batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=options.num_workers,
                                 pin_memory=True)

    optimizer = build_optimizer(model_without_ddp, options)
    lr_scheduler = build_scheduler(optimizer, options)
    # optimizer = create_optimizer(options, model_without_ddp)
    # lr_scheduler, _ = create_scheduler(options, optimizer)

    if options.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[options.gpu])
        model_without_ddp = model.module

    if options.pretrain_from:
        checkpoint = torch.load(options.pretrain_from, map_location='cpu')
        if 'model' in checkpoint:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    if options.resume_from:
        checkpoint = torch.load(options.resume_from, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if (not options.eval) and 'optimizer' in checkpoint \
                and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.step(lr_scheduler.last_epoch)
            options.start_epoch = checkpoint['epoch'] + 1
            if utils.is_main_process():
                summary_writer.iter_num = checkpoint['iter_num']
            print('resume optimizer')
        print('resume finished.')

    if options.eval:
        test_stats = evaluate(model, evaluator, data_loader_val, device)
        test_info = 'Test on ' + options.val_dataset
        for k, v in test_stats.items():
            test_info += ' %s:%.4f' % (k, v)
        print(test_info)
        return

    print("Start training")
    log_dir = Path(options.log_dir)
    start_time = time.time()
    for epoch in range(options.start_epoch, options.num_epochs):
        if options.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, options, summary_writer)
        lr_scheduler.step()

        if options.log_dir and utils.is_main_process():
            if (epoch + 1) % options.save_freq == 0:
                if not os.path.exists(options.log_dir):
                    os.makedirs(options.log_dir, exist_ok=True)
                checkpoint_path = log_dir / f'checkpoints/checkpoint{epoch:04}.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'options': options,
                    'iter_num': summary_writer.iter_num,
                }, checkpoint_path, _use_new_zipfile_serialization=False)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if options.log_dir and utils.is_main_process():
            with (log_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if (epoch + 1) % options.eval_freq == 0:
            test_stats = evaluate(model, evaluator, data_loader_val, device)
            test_info = 'Test on ' + options.val_dataset
            for k, v in test_stats.items():
                test_info += ' %s:%.4f' % (k, v)
            print(test_info)

            if options.log_dir and utils.is_main_process():
                with (log_dir / "log.txt").open("a") as f:
                    f.write(test_info + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    options = DDPTrainOptions().parse_args()
    main(options)



import datetime
import json
import logging
import os
import random
import time

import numpy as np
import omegaconf
import timm
import torch
import torch.nn.functional as F
from torch import distributed
from torch.backends import cudnn
from torch.utils import data as torchdata

from src.routines.registry import get_routine
import src.utils.misc as misc
from src.datasets.registry import create_dataset
from src.models import create_model
from src.utils.lars import LARS
from src.utils.wandb import init_wandb

assert timm.__version__ in ["0.3.2", "0.4.12"] # version check
import timm.optim.optim_factory as optim_factory

log = logging.getLogger(__name__)

def main(args):
    if args.distributed and distributed.is_available():
        world_size, rank = misc.init_distributed()
    else:
        world_size, rank = 1, 0
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints"); os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = os.path.join(args.experiment_dir, "saves"); os.makedirs(save_dir, exist_ok=True)
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    log.info("{}".format(resolved_args).replace(', ', ',\n'))
    if rank == 0 and args.wandb:
        name = (args.name + "//" + args.sub_name) if hasattr(args, "sub_name") else (".LOCAL" + "//" + args.name)
        group = None
        init_wandb(resolved_args,
                   name=name, group=group, dir=args.experiment_dir,
                   project='poodle' if not hasattr(args, "wandb_project") else args.wandb_project)

    gpu = rank % torch.cuda.device_count()
    device = torch.device(gpu)
    is_dist = misc.is_dist_avail_and_initialized()
    if not hasattr(args, 'num_workers') or args.num_workers is None:
        num_workers = int(len(os.sched_getaffinity(0))) # num CPUs on the machine
        if distributed.is_initialized():
            num_workers = int(num_workers/torch.cuda.device_count())
    else:
        num_workers = args.num_workers
    log.info(f"Data loader num workers: {num_workers}")
    repeat_sample = args.dataset_configs.repeat_sample if hasattr(args.dataset_configs, "repeat_sample") else None
    assert repeat_sample is None or repeat_sample > 0, "repeat_sample should be None or a positive integer"
    log.info(f"Repeat sample is: {repeat_sample}")
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.get('benchmark', True)
    
    # -- Dataset --
    train_dataset = create_dataset(args.dataset, split="train", **args.dataset_configs)
    log.info(f'Number of train samples: {len(train_dataset)}')
    if is_dist:
        train_sampler = torchdata.DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = torchdata.RandomSampler(train_dataset)
    pin_memory = True if not hasattr(args, "pin_memory") else args.pin_memory
    train_dataloader = torchdata.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                            batch_size=args.batch_size,
                                            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
                                            persistent_workers=False, prefetch_factor=3 if num_workers != 0 else None)
 
    # -- Model --
    model_configs = dict(args.model_configs) if args.model_configs is not None else {}
    model = create_model(args.model, **model_configs)
    model.to(device)
    if is_dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    log.info(f"Model = {model}")
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log.info(f"Number of trainable parameters: {total_params}")
    decoder_params = sum(param.numel() for name, param in model.named_parameters() if param.requires_grad and "decoder" in name)
    log.info(f"Number of trainable decoder parameters: {decoder_params}")
    model_without_ddp, model = model, torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu]) if is_dist else model
    
    # -- Optimizer --
    eff_batch_size = args.batch_size * args.accum_iter * world_size * (1 if repeat_sample is None else repeat_sample) * (1 if not hasattr(args.dataset_configs, "block_sample") else args.dataset_configs.block_sample)
    if args.lr is None:
        assert args.blr is not None
        args.lr = args.blr * eff_batch_size / 256
    else:
        args.blr = args.blr
        args.lr = args.lr
    log.info(f"LR: base_lr {args.blr} | eff_batch_size {eff_batch_size} | actual_lr {args.lr}")
    
    if hasattr(args, 'optimizer') and args.optimizer == 'lars':
        optimizer = LARS(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8 if args.fp16 else 1e-6)
    log.info(f"Optimizer: {optimizer}")
    loss_scaler = misc.NativeScalerWithGradNormCount(fp16=args.fp16)
    
    # -- Retrieve routine functions --
    train_routine = get_routine(args.train_routine)
    
    # -- Checkpointing --
    resume = args.resume if hasattr(args, "resume") else None
    resume_training = False
    if not resume:
        resume = checkpoint_dir
        resume_training = True
    start_epoch = misc.load_model(resume=resume, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, resume_training=resume_training)

    # -- Train --
    log.info(f"Starting training at epoch {start_epoch}")
    print_freq = args.print_freq if hasattr(args, "print_freq") else 50
    accum_iter = args.accum_iter if hasattr(args, "accum_iter") else 1
    start_time = time.time()
    epoch_increment = 1 if repeat_sample is None else repeat_sample
    for epoch in range(start_epoch, args.epochs, epoch_increment):
        if is_dist:
            train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_routine(
            epoch=epoch,
            model=model,
            data_loader=train_dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            device=device,
            args=args,
            print_freq=print_freq,
            accum_iter=accum_iter
        )
        torch.cuda.synchronize()
        post_epoch = epoch + epoch_increment
        if rank == 0:
            if post_epoch % args.checkpoint_freq == 0 or post_epoch >= args.epochs:
                misc.save_model(checkpoint_dir, args=args, epoch=post_epoch, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': post_epoch,}
            with open(os.path.join(args.experiment_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if is_dist:
            torch.distributed.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}'.format(total_time_str))
    log.info("Training done.")

if __name__ == '__main__':
    pass
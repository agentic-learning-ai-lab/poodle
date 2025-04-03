
import datetime
import json
import logging
import math
import os
import random
import copy
import sys
import time

import numpy as np
import omegaconf
import timm
import torch
from torch import distributed
from torch.backends import cudnn
from torch.utils import data as torchdata

import src.utils.lr_sched as lr_sched
import src.utils.misc as misc
from src.datasets.registry import create_dataset
from src.models import create_model
from src.utils.wandb import init_wandb, wandb_log
from src.utils.visualize import get_mixed_batch, visualize_predictions
from src.utils.flow import compute_occlusion_mask
from src.utils.losses import EPE

assert timm.__version__ in ["0.3.2", "0.4.12"] # version check
import timm.optim.optim_factory as optim_factory

log = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(epoch, model, num_workers, is_dist, world_size, rank, device, args, print_freq=50):
    model.eval()
    
    eval_dataset = create_dataset(args.eval_dataset, **args.eval_dataset_configs)
    eval_sampler = (torchdata.DistributedSampler(dataset=eval_dataset, num_replicas=world_size, rank=rank, shuffle=True) 
                    if is_dist else torchdata.RandomSampler(eval_dataset))
    dataloader = torchdata.DataLoader(dataset=eval_dataset, sampler=eval_sampler,
                                      batch_size=args.eval_batch_size, 
                                      num_workers=num_workers, pin_memory=True, drop_last=True,
                                      persistent_workers=True, prefetch_factor=1)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = '[EVALUATE] Epoch: [{}]'.format(epoch)
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.bfloat16 if args.get('bfloat', False) else torch.float16):
            flow = model(source=batch['source'], target=batch['target'])
            epe = EPE(flow, batch['gt_flow'], batch['gt_flow_mask'], sparse=True, mean=True)
            loss_dict = {"supervised_EPE": epe}
            
        torch.cuda.synchronize()

        metric_logger.update(**loss_dict)
        
        for k in loss_dict:
            loss_dict[k] = misc.all_reduce_mean(loss_dict[k], device)
    
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    meter_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_dict = {**dict(filter(lambda x: "supervised" in x[0], meter_dict.items())), "epoch": epoch}
    if rank == 0:
        wandb_log({f"eval/{k}": v for (k, v) in eval_dict.items()}, commit=False)
        eval_dict = {**{f'eval_{k}': v for k, v in eval_dict.items()}, 'epoch': epoch,}
        with open(os.path.join(args.experiment_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(meter_dict) + "\n")
    return eval_dict

@torch.no_grad()
def visualize_batches(epoch, save_dir, model, batches,
                      is_dist, device, args,):
    model.eval()
    
    for name, batch in batches.items():
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        source, target = batch['source'], batch['target']
        gt_flow, gt_flow_mask = batch.get('gt_flow', torch.zeros_like(source[:, :2])), batch.get('gt_flow_mask', torch.ones_like(source[:, :1]))
        with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.bfloat16 if args.get('bfloat', False) else torch.float16):
            pred_flow, p1, p2 = model(source, target, return_all_scales=True, return_features=True)
            pred_backward_flow = model(target, source, return_all_scales=True)
            occlusion_mask, flow_diff = compute_occlusion_mask(pred_flow[0], pred_backward_flow[0], model.module.alpha1 if is_dist else model.alpha1, model.module.alpha2 if is_dist else model.alpha2)
            
        fp = os.path.join(save_dir, f"{epoch}-{name}")
        visualize_predictions(fp, source, target, gt_flow, gt_flow_mask,
                              pred_flow, pred_backward_flow, p1, p2,
                              occlusion_mask, flow_diff,
                              scale_to_gt=hasattr(batch, 'gt_flow'),
                              rescale=args.get('vis_rescale', 0.3))

def train_one_epoch(epoch,
                    model, model_without_ddp,
                    dataloader, optimizer, loss_scaler,
                    device, args,
                    is_dist,
                    print_freq=50):
    model.train()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    repeat_sample = args.dataset_configs.get('repeat_sample', None)
    step_per_iter = 1 if repeat_sample is None else repeat_sample

    use_occ_masking = epoch >= args.get('occ_start_epochs', 0)
    use_selfsup = epoch >= args.get('selfsup_start_epochs', float('inf'))
    use_ssl = epoch >= args.get('ssl_start_epochs', float('inf'))
    if use_ssl:
        params = model_without_ddp.enable_ssl()
        if params is not None:
            optimizer.add_param_group({'params': params[0], 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': params[1], 'weight_decay': 0.})
    if is_dist:
        torch.distributed.barrier()
    
    if epoch > 0 and hasattr(args, "dt_curriculum") and args.dt_curriculum:
        dt_increment_freq = int(args.dt_curriculum.split('-')[0])
        if epoch % dt_increment_freq == 0:
            dt_increment = int(args.dt_curriculum.split('-')[1])
            log.info(f"Updating delta_t from {dataloader.dataset.delta_t} to {dataloader.dataset.delta_t + dt_increment}")
            dataloader.dataset.delta_t += dt_increment
    
    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if repeat_sample is not None:
            batch = {k: v.flatten(0, 1) for k, v in batch.items()}

        global_step = step_per_iter*data_iter_step + epoch*len(dataloader)
        global_epoch = step_per_iter * data_iter_step / len(dataloader) + epoch
        lr_sched.adjust_learning_rate(optimizer, global_epoch, args)
        
        loss_weights = copy.deepcopy(dict(args.loss_weights))
        if args.get('selfsup_warmup_epochs', 0) > 0:
            loss_weights['selfsup'] *= min(1., max(0., global_epoch - args.selfsup_start_epochs)/(args.selfsup_warmup_epochs))
        if args.get('ssl_warmup_epochs', 0) > 0:
            ssl_weight = min(1., max(0., global_epoch - args.ssl_start_epochs)/(args.ssl_warmup_epochs))
            loss_weights['ssl'] *= ssl_weight
        if hasattr(args, 'base_momentum'):
            momentum = 1 - (1 - args.base_momentum) * (math.cos(math.pi * max(0., global_epoch - args.ssl_start_epochs) / (args.epochs - args.ssl_start_epochs)) + 1) / 2
        else:
            momentum = None
            
        with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.bfloat16 if args.get('bfloat', False) else torch.float16):
            loss, loss_dict, raw_loss_dict, metric_dict = model(**batch, return_loss=True, loss_weights=loss_weights, use_occ_masking=use_occ_masking, use_selfsup=use_selfsup, use_ssl=use_ssl, momentum=momentum)

        raw_loss_dict = {f'raw_{k}': v for k, v in raw_loss_dict.items()}
        
        if not math.isfinite(loss.item()):
            log.error("Loss is {}, stopping training.".format(loss.item()))
            for loss_name, loss_val in loss_dict.items():
                if not math.isfinite(loss_val):
                    log.error("Loss={} is {}, stopping training".format(loss_name, loss_val))
            if misc.get_rank() == 0:
                wandb_log({"train/loss_weights": loss_weights,
                           "train/loss": loss_dict,
                           "train/raw_loss": raw_loss_dict,
                           "train/metric": metric_dict,
                           "train/epoch": int(global_epoch),
                           "train/step": global_step},
                          step=global_step)
            sys.exit(1)
            
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=True,
                                clip_grad=args.clip_grad)
        metric_dict['grad_norm'] = torch.nan_to_num(torch.round(grad_norm, decimals=4), 0.)
        optimizer.zero_grad()
            
        torch.cuda.synchronize()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(**loss_dict)
        metric_logger.update(**raw_loss_dict)
        metric_logger.update(**metric_dict)
        
        for k in loss_dict:
            loss_dict[k] = misc.all_reduce_mean(loss_dict[k], device)
        for k in metric_dict:
            metric_dict[k] = misc.all_reduce_mean(metric_dict[k], device)
        for k in raw_loss_dict:
            raw_loss_dict[k] = misc.all_reduce_mean(raw_loss_dict[k], device)

        if data_iter_step % print_freq == 0 and misc.get_rank() == 0:
            wandb_log({"train/loss": loss_dict, 
                       "train/raw_loss": raw_loss_dict,
                       "train/metric": metric_dict, 
                       "train/lr": lr, 
                       "train/epoch": epoch,
                       "train/step": (epoch*len(dataloader)) + data_iter_step},
                      step=(epoch*len(dataloader)) + (data_iter_step * step_per_iter))

    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train(args, experiment_dir, checkpoint_dir, save_dir, num_workers, gpu, device, is_dist, world_size, rank):
    # -- dataset --
    train_dataset = create_dataset(args.dataset, **args.dataset_configs)
    if is_dist:
        train_sampler = torchdata.DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = torchdata.RandomSampler(train_dataset)
    train_dataloader = torchdata.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                            batch_size=args.batch_size, 
                                            num_workers=num_workers, pin_memory=True, drop_last=True,
                                            persistent_workers=True, prefetch_factor=args.get('prefetch_factor', 1))
    repeat_sample = args.dataset_configs.get('repeat_sample', None)
    assert repeat_sample is None or repeat_sample > 0, "repeat_sample should be None or a positive integer"
    log.info(f"Repeat sample is: {repeat_sample}")
    
    # -- model -- 
    model = create_model(args.model, **args.model_configs)
    log.info(model)
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    log.info(f"Number of trainable parameters: {total_params}")
    model.to(device)
    model_without_ddp = model
    if is_dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # -- optimizer --
    eff_batch_size = args.batch_size * world_size
    if args.lr is None:
        assert args.blr is not None
        raise NotImplementedError("LR scaling not yet support")
    else:
        lr = args.lr
    log.info(f"LR: base_lr {args.blr} lr {args.lr} | eff_batch_size {eff_batch_size} | actual_lr {lr}")
    
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    opt_name = args.get('optimizer', 'adam')
    if opt_name == 'adam':
        assert args.weight_decay == 0.
        optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=lr)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        
    log.info(f"Optimizer, {optimizer}")
    loss_scaler = misc.NativeScalerWithGradNormCount(fp16=args.fp16)
    
    # -- Checkpointing -- 
    resume = args.resume if hasattr(args, "resume") else None
    if resume:
        start_epoch = misc.load_model(resume=resume,
                                      model_without_ddp=model_without_ddp,
                                      resume_training=False)
    if not resume:
        resume = checkpoint_dir
        start_epoch = misc.load_model(resume=resume,
                                    model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                                    resume_training=True)
    
    # -- Initial Eval and Visualization -- 
    if rank == 0 and args.vis:
        assert args.eval_dataset == 'kitti'
        eval_dataset_configs = {**args.eval_dataset_configs, **{"img_size": (370, 1224)}}
        vis_datasets = {"eval": create_dataset(args.eval_dataset, **eval_dataset_configs)}
        if hasattr(args, 'vis_dataset') and args.vis_dataset is not None: 
            vis_dataset_configs = omegaconf.OmegaConf.to_container(args.vis_dataset_configs, resolve=True, throw_on_missing=True)
            if hasattr(args, 'vis_dataset_delta_ts') and args.vis_dataset_delta_ts is not None:
                delta_ts = args.vis_dataset_delta_ts
            else:
                delta_ts = [vis_dataset_configs['delta_t']]
            for delta_t in delta_ts:
                vis_dataset_configs['delta_t'] = delta_t
                vis_datasets[f'vis_dt{delta_t}'] = create_dataset(args.vis_dataset, **vis_dataset_configs)
        vis_batches = {k: get_mixed_batch(v) for k, v in vis_datasets.items()}
        flow_model = None
        for k, v in vis_batches.items():
            if v.get('gt_flow', None) is None:
                if flow_model is None:
                    flow_model = create_model("raft", **args.vis_model_configs)
                    flow_model = flow_model.to(device)
                source, target = v['source'].to(device), v['target'].to(device)
                flow = flow_model(source, target, iters=24, test_mode=True)[1]
                vis_batches[k]['gt_flow'] = flow
    if start_epoch == 0:
        if args.eval:
            evaluate(0, model, num_workers, is_dist, world_size, rank, device, args, print_freq=50)
        # call eval routine here
        if rank == 0 and args.vis:
            visualize_batches(0, save_dir, model, vis_batches, is_dist, device, args)
            
    # -- train --
    log.info(f"Starting training at epoch {start_epoch}")
    start_time = time.time()
    epoch_increment = 1 if repeat_sample is None else repeat_sample
    for epoch in range(start_epoch, args.epochs, epoch_increment):            
        if is_dist:
            train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(epoch,
                                      model, model_without_ddp,
                                      train_dataloader, optimizer, loss_scaler,
                                      device, args,
                                      is_dist,
                                      args.print_freq)
        
        torch.cuda.synchronize()
        post_epoch = epoch + epoch_increment
        # -- run eval before train in iterator; same as using post_epoch --
        if args.eval and post_epoch % args.eval_freq == 0:
            evaluate(post_epoch, model, num_workers, is_dist, world_size, rank, device, args, print_freq=50)
        # -- run vis/checkpointing before train in iterator --
        if rank == 0:
            if post_epoch and post_epoch % args.checkpoint_freq == 0:
                misc.save_model(checkpoint_dir, args=args, epoch=post_epoch, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler)
            if (post_epoch % args.vis_freq == 0) and args.vis:
                visualize_batches(post_epoch, save_dir, model, vis_batches, is_dist, device, args)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': post_epoch,}
            with open(os.path.join(args.experiment_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if is_dist:
            torch.distributed.barrier()
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}'.format(total_time_str))
    log.info("Training done.")
        
def main(args):
    # -- setup distributed --
    if args.distributed and distributed.is_available():
        world_size, rank = misc.init_distributed()
    else:
        world_size, rank = 1, 0
        
    # -- setup directories --
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints"); os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = os.path.join(args.experiment_dir, "saves"); os.makedirs(save_dir, exist_ok=True)
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    log.info("{}".format(resolved_args).replace(', ', ',\n'))
    
    # -- wandb --
    if rank == 0 and args.wandb:
        name = (args.name + "//" + args.sub_name) if hasattr(args, "sub_name") else (".LOCAL" + "//" + args.name)
        group = None
        init_wandb(resolved_args,
                   name=name, group=group, dir=args.experiment_dir,
                   project="poodle" if not hasattr(args, "wandb_project") else args.wandb_project)

    # -- gpu/device, num_workers(cpus) --
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
    
    # -- tf32 --
    log.info("TF32 matmul: {}".format(torch.backends.cuda.matmul.allow_tf32))
    log.info("TF32 cudnn: {}".format(torch.backends.cudnn.allow_tf32))

    # -- random seeding, benchmarking --
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.benchmark:
        cudnn.benchmark = True
    
    train(args, args.experiment_dir, checkpoint_dir, save_dir, num_workers, gpu, device, is_dist, world_size, rank)

import copy
import logging
import math
import sys

import timm
import torch
from torch.utils import data as torchdata

import src.utils.lr_sched as lr_sched
import src.utils.misc as misc
from src.routines.registry import register_routine
from src.utils.losses import weigh_losses
from src.utils.wandb import wandb_log

assert timm.__version__ in ["0.3.2", "0.4.12"] # version check

log = logging.getLogger(__name__)

@register_routine
@torch.no_grad()
def poodle_evaluate(epoch: int,
                    step: int,
                    model: torch.nn.Module,
                    data_loader: torchdata.DataLoader,
                    device: torch.device,
                    dataset_name,
                    fp16,
                    loss_weights,
                    print_freq: int=10):
    return {}

@register_routine
def poodle_train_one_epoch(epoch: int,
                          model: torch.nn.Module,
                          data_loader: torchdata.DataLoader,
                          optimizer: torch.optim.Optimizer,
                          loss_scaler: misc.NativeScalerWithGradNormCount,
                          device: torch.device,
                          args,
                          print_freq: int=50,
                          accum_iter=1):
    model.train(True)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    repeat_sample = args.dataset_configs.repeat_sample if hasattr(args.dataset_configs, "repeat_sample") else None
    step_per_iter = 1 if repeat_sample is None else repeat_sample
    
    loss_weights = copy.deepcopy(dict(args.loss_weights))
    
    optimizer.zero_grad()
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [x.to(device, non_blocking=True) if (isinstance(x, torch.Tensor)) else x for x in batch]
        if (hasattr(args.dataset_configs, "repeat_sample") and args.dataset_configs.repeat_sample is not None):
            batch = [x.flatten(0, 1) for x in batch]
 
        step = step_per_iter*data_iter_step/len(data_loader) + epoch
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step, args)
 
        with torch.cuda.amp.autocast(enabled=args.fp16):
            # adjust momentum based on cosine schedule
            base_momentum = args.base_momentum
            momentum = 1 - (1 - base_momentum) * (math.cos(math.pi * step / args.epochs) + 1) / 2
            loss_dict, metric_dict = model(*batch, momentum=momentum)
            
        weighted_loss_dict = weigh_losses(loss_dict, loss_weights)
        loss = sum(weighted_loss_dict.values())
        weighted_loss_dict["loss"] = loss

        if not torch.all(torch.isfinite(loss)):
            log.error("Loss is {}, stopping training.".format(loss.item()))
            for loss_name, loss_val in loss_dict.items():
                if not math.isfinite(loss_val):
                    log.error("Loss={} is {}, stopping training".format(loss_name, loss_val))
            if misc.get_rank() == 0:
                wandb_log({"train/raw_loss": loss_dict, "train/metric": metric_dict, "train/epoch": epoch, "train/step": (epoch*len(data_loader)) + data_iter_step},
                          step=(epoch*len(data_loader)) + (data_iter_step * step_per_iter))
            sys.exit(1)
        
        loss /= accum_iter
        update_grad = (data_iter_step+1) % accum_iter == 0
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=update_grad,
                                clip_grad=args.clip_grad)
        if update_grad:
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss.item())
        metric_logger.update(**weighted_loss_dict)
        metric_logger.update(**metric_dict)
        metric_logger.update(grad_norm=grad_norm)
        
        for k in loss_dict:
            loss_dict[k] = misc.all_reduce_mean(loss_dict[k], device)
        for k in loss_dict:
            weighted_loss_dict[k] = misc.all_reduce_mean(weighted_loss_dict[k], device)
        for k in metric_dict:
            metric_dict[k] = misc.all_reduce_mean(metric_dict[k], device)
            
        if data_iter_step % 10 == 0 and misc.get_rank() == 0:
            wandb_log({"train/weighted_loss": weighted_loss_dict,
                       "train/raw_loss": loss_dict, 
                       "train/metric": metric_dict, 
                       "train/lr": lr, 
                       "train/epoch": epoch,
                       "train/grad_norm": grad_norm,
                       "train/step": (epoch*len(data_loader)) + data_iter_step},
                      step=(epoch*len(data_loader)) + (data_iter_step * step_per_iter))
            
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@register_routine
@torch.no_grad()
def poodle_visualize_batch(filename,
                           dataset,
                           dataset_name,
                           args,
                           save_dir,
                           batch,
                           model,
                           device,
                           rescale=1.0):
    return

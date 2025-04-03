# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import logging
import os
import pickle
import resource
import shutil
import signal
import sys
import tempfile
import time
from collections import defaultdict, deque
from multiprocessing.context import ForkContext
from pathlib import Path

import torch
import torch.distributed as dist
try:
    from torch._six import inf
except:
    from torch import inf

log = logging.getLogger(__name__)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t /= get_world_size()
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        torch.cuda.reset_peak_memory_stats()
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{value:.3f}({avg:.3f})', window_size=print_freq)
        data_time = SmoothedValue(fmt='{value:.3f}({avg:.3f})', window_size=print_freq)
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg.append('max cpu mem: {cpu_memory:.0f}')
        if torch.cuda.is_available():
            log_msg.append('max gpu mem: {gpu_memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        KB = 1024.0
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                iter_time.synchronize_between_processes()
                data_time.synchronize_between_processes()
                if torch.cuda.is_available():
                    log.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        cpu_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB,
                        gpu_memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    log.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        cpu_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class MultiloaderMetricLogger(MetricLogger):
    def log_every(self, iterables, print_freq, header=None):
        len_iterables = sum([len(iterables[key]) for key in iterables])
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{value:.3f}({avg:.3f})', window_size=print_freq)
        data_time = SmoothedValue(fmt='{value:.3f}({avg:.3f})', window_size=print_freq)
        space_fmt = ':' + str(len(str(len_iterables))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg.append('max cpu mem: {cpu_memory:.0f}')
        if torch.cuda.is_available():
            log_msg.append('max gpu mem: {gpu_memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        KB = 1024.0
        MB = 1024.0 * 1024.0
        iterables = {ds_name: iter(iterable) for ds_name, iterable in iterables.items()}
        while True:
            found = False
            for ds_name, iterable in iterables.items():
                try:
                    obj = next(iterable)
                    found = True
                except:
                    continue
                data_time.update(time.time() - end)
                yield ds_name, obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == len_iterables - 1:
                    eta_seconds = iter_time.global_avg * (len_iterables - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        log.info(log_msg.format(
                            i, len_iterables, eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            cpu_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB,
                            gpu_memory=torch.cuda.max_memory_allocated() / MB))
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        log.info(log_msg.format(
                            i, len_iterables, eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            cpu_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB))
                i += 1
                end = time.time()
            if not found:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iterables))

def suppress_prints(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    
def suppress_logging(is_master):
    if not is_master:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.WARN)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def cleanup_dist(args):
    log.info("Destroying process group.")
    if os.path.isfile(args.dist_url):
        os.remove(args.dist_url)
    dist.destroy_process_group()
    
def init_distributed():

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = None, None

    if (rank is None) or (world_size is None):
        try:
            if "SLURM_NNODES" in os.environ and "SLURM_PROCID" in os.environ and "SLURM_TASKS_PER_NODE" in os.environ:
                world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
                rank = int(os.environ['SLURM_PROCID'])
            elif "WORLD_SIZE" in os.environ and "RANK" in os.environ:
                world_size = int(os.environ["WORLD_SIZE"])
                rank = int(os.environ['RANK'])
            gpu = rank % torch.cuda.device_count()
            if ("MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ):
                dist_url = "tcp://{}:{}".format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
            else:
                dist_url="tcp://localhost:40000"
        except Exception:
            log.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        log.info('| distributed init (rank {}): {}, gpu {}, world_size {}'.format(rank, dist_url, gpu, world_size))
        dist.init_process_group(
            init_method=dist_url,
            backend='nccl',
            world_size=world_size,
            rank=rank)
        torch.cuda.set_device(gpu)
        # suppress logging only if init lol
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARN)
    except Exception as e:
        world_size, rank = 1, 0
        log.info(f'distributed training not available {e}')
    
    return world_size, rank


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank = get_rank()
    world_size = get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            os.makedirs('/tmp/dist_test', exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir='/tmp/dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)
    # dump the part result to the dir
    tmp_file = os.path.join(tmpdir, f'part_{rank}.pkl')
    pickle.dump(result_part, open(str(tmp_file), "wb"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_result = pickle.load(open(str(part_file), "rb"))
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp16: bool):
        self._scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(output_dir, args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()

        save_on_master(to_save, checkpoint_path)

def load_model(resume, model_without_ddp, optimizer=None, loss_scaler=None, resume_training=False):
    if os.path.isdir(resume):
        resume = get_last_checkpoint(resume)
    if resume:
        logging.info(f"Checkpoint found. Loading from: {resume}")
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=resume_training)
        log.info(f"Checkpoint loading message:\n missing_keys:{msg.missing_keys} \n unexpected_keys:{msg.unexpected_keys}")
        if resume_training:
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                log.info(f"With start epoch: {start_epoch}.")
            if optimizer is not None and 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                log.info("With optim.")
            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                log.info("With loss scheduler.")
            return start_epoch
        return 0
    else:
        logging.info("No checkpoint found.")
        return 0

def eval_load_model(resume, model_without_ddp, optimizer=None, loss_scaler=None, resume_training=False, load_final_norm=True):
    if os.path.isdir(resume):
        resume = get_last_checkpoint(resume)
    if resume:
        logging.info(f"Checkpoint found. Loading from: {resume}")
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        # HACK (ch3451): for backward compatibility EMAEvX --> MAE
        # reshape_keys = ['mask_token', 'decoder_pos_embed']
        # for key in reshape_keys:
        #     if hasattr(model_without_ddp, key) and key in checkpoint['model']:
        #         checkpoint['model'][key] = checkpoint['model'][key].reshape(getattr(model_without_ddp, key).shape)
        # HACK (ch3451): for backward compatibility motion-ssl --> mjepa
        backbone_keys = list(key for key in checkpoint['model'].keys() if 'backbone.' in key)
        for key in backbone_keys:
            checkpoint['model'][key.replace('backbone.', '')] = checkpoint['model'].pop(key)
        if not load_final_norm and 'norm.weight' in checkpoint['model']:
            # HACK (ch3451): do not load norm weights
            del checkpoint["model"]["norm.weight"]
            del checkpoint["model"]["norm.bias"]
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=resume_training)
        log.info(f"Checkpoint loading message:\n missing_keys:{msg.missing_keys} \n unexpected_keys:{msg.unexpected_keys}")
        if resume_training:
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                log.info(f"With start epoch: {start_epoch}.")
            if optimizer is not None and 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                log.info("With optim.")
            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                log.info("With loss scheduler.")
            return start_epoch
        return 0
    else:
        logging.info("No checkpoint found.")
        return 0

def get_last_checkpoint(d):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        d (string): checkpoint directory of the current job.
    """
    names = [os.path.join(d, x) for x in os.listdir(d)] if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        log.info("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]
        return os.path.join(d, name)


def all_reduce_mean(x, device):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).to(device)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def log_gpu_memory_usage(section='', reset=True):
    if torch.cuda.is_available():
        log.info(f"Max GPU memory used for {section}: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    if reset:
        torch.cuda.reset_peak_memory_stats()

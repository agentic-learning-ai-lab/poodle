import sys; import os; sys.path.append(os.getcwd())
from bisect import bisect
import gc
import glob
import os
from typing import Callable, List

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tvF
from decord import VideoReader, cpu
from torch.utils import data as torchdata
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader as TVVideoReader
from torchvision.transforms import InterpolationMode

from src.datasets.registry import register_dataset
from src.utils.affine import AffineWithGrid
from src.utils.ffprobe import ffprobe
from src.utils.misc import get_rank

import logging 
log = logging.getLogger(__name__)

def round_up_odd(x):
    return int(np.ceil(x / 2) * 2 + 1)

@register_dataset
def walking_tours(data_dir, split,
                  img_size=None,
                  dataset_repeats=None,
                  delta_t=[15, 15],
                  frames_per_clip=300,
                  repeat_sample=1,
                  initial_size=[1080, 1920],
                  initial_scale=[0.75, 1.25],
                  initial_crop_size=[512, 1024],
                  color_distortion_strength=1.0,
                  norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                  subset=1,
                  backend='decord'):
    assert subset <= 1.0
    assert split == 'train', 'Only train split is supported'
    assert color_distortion_strength <= 1.0
    assert backend in ['decord', 'torchvision', 'torchvision-videoreader']

    # transforms
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    if img_size is not None:
        resize = transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=None)
    if initial_crop_size is not None:
        random_crop = transforms.RandomCrop(initial_crop_size)
    
    # data augmentations
    s = color_distortion_strength
    h, w = img_size if img_size is not None else initial_crop_size
    gaussian_kernel = (round_up_odd(0.1*h), round_up_odd(0.1 * w))
    data_aug = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(gaussian_kernel, sigma=(0.1, 2.0))], p=0.5),
        normalize
    ])
    
    def aug_fn(img1: torch.Tensor, img2: torch.Tensor):
        img1 = tvF.to_tensor(img1)
        img2 = tvF.to_tensor(img2)

        if initial_scale is not None:
            assert initial_size is not None, 'initial_size must be provided if initial_scale is provided'
            scale = float(torch.empty(1).uniform_(initial_scale[0], initial_scale[1]).item())
            img1, img2 = torch.unbind(
                tvF.resize(torch.stack([img1, img2], dim=0), (int(initial_size[0]*scale), int(initial_size[1]*scale)), interpolation=InterpolationMode.BILINEAR, antialias=None),
                dim=0
            )

        if initial_crop_size is not None:
            img1, img2 = torch.unbind(
                random_crop(torch.stack([img1, img2], dim=0)),
                dim=0
            )
        
        if img_size is not None:
            img1, img2 = [resize(x) for x in [img1, img2]]
        
        img1_ = data_aug(img1)
        img2_ = data_aug(img2)
        
        return img1_, img2_, img1, img2

    dataset = WalkingToursDataset(
        data_dir=data_dir,
        dataset_repeats=dataset_repeats,
        aug_fn=aug_fn,
        mode=split,
        delta_t=delta_t,
        norm_mean=norm_mean,
        norm_std=norm_std,
        frames_per_clip=frames_per_clip,
        repeat_sample=repeat_sample,
        backend=backend,
    )

    if subset != 1.0:
        n = int(len(dataset) * subset)
        data = torchdata.random_split(data, (n, len(data)-n), generator=torch.Generator().manual_seed(1))[0]

    return dataset


class WalkingToursDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 aug_fn: Callable,
                 mode: str = "train",
                 delta_t: List[int] = [3, 15],
                 norm_mean: List[float] = [0.485, 0.456, 0.406],
                 norm_std: List[float] = [0.229, 0.224, 0.225],
                 frames_per_clip: int = 300,
                 repeat_sample: int = 1,
                 backend='decord',
                 dataset_repeats=None,
                 ):
        
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.aug_fn = aug_fn
        self.mode = mode
        self.delta_t = delta_t
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.frames_per_clip = frames_per_clip
        self.repeat_sample = repeat_sample
        self.backend = backend
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_paths = glob.glob(os.path.join(data_dir, '**', "*.mp4"), recursive=True)

        self.video_paths = video_paths
        self.video_clip_idx_start = []
        self.video_readers = []
        self.video_md = []
        self.video_rotations = []
        cur_idx = 0
        for video_path in video_paths:
            _, vr_len, vr_md, vr_rotation = self.create_video_reader(video_path, 0)
            self.video_clip_idx_start.append(cur_idx)
            cur_idx += (vr_len // self.frames_per_clip)
            self.video_readers.append({})
            self.video_md.append(vr_md)
            self.video_rotations.append(vr_rotation)
        
        self._dataset_len = cur_idx
        
        self.dataset_repeats = dataset_repeats
        if dataset_repeats is not None:
            self._dataset_len *= dataset_repeats
    
    def get_video_reader(self, clip_idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        video_idx = bisect(self.video_clip_idx_start, clip_idx) - 1

        if cpuid not in self.video_readers[video_idx]:
            self.video_readers[video_idx][cpuid] = self.create_video_reader(self.video_paths[video_idx], cpuid)[0]
        return self.video_readers[video_idx][cpuid], self.video_clip_idx_start[video_idx], self.video_md[video_idx], self.video_rotations[video_idx]
    
    def create_video_reader(self, video_path, cpuid):
        if self.backend == 'decord':
            vr = VideoReader(video_path, num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
            vr_md = None
            vr_rotation = None
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(video_path, "video")
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            try:
                vr_rotation = -int(ffprobe(video_path).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
            except:
                vr_rotation = 0
            vr_md = vr.get_metadata()['video']
            vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1
        return vr, vr_len, vr_md, vr_rotation

    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, clip_idx):
        if self.dataset_repeats is not None:
            clip_idx = clip_idx % (self._dataset_len // self.dataset_repeats)
        vr, clip_idx_start, vr_md, vr_rotation = self.get_video_reader(clip_idx)
        start_idx = (clip_idx - clip_idx_start) * self.frames_per_clip
        end_idx = start_idx + self.frames_per_clip - self.delta_t[1]
        i_s = np.random.randint(start_idx, end_idx, size=self.repeat_sample)
        delta_ts = np.random.randint(self.delta_t[0], self.delta_t[1]+1, size=self.repeat_sample)
        i_s = np.array([index for i, delta_t in zip(i_s, delta_ts) for index in [i, i+delta_t]])
        sort_indexes = np.argsort(i_s).astype(np.int32)
        unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
        if self.backend == 'decord':
            imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
            vr.seek(0)
        elif 'torchvision' in self.backend:
            res = []
            i_s_ = [x / vr_md['fps'][0] for x in i_s[sort_indexes]]
            for i_ in i_s_:
                vr.seek(i_)
                count = 0
                while True:
                    try:
                        res.append(next(vr)['data'])
                        break
                    except StopIteration:
                        log.warning(f"StopIteration at {i_ + (count * 1/vr_md['fps'][0])} for {self.video_paths[idx]}")
                        count += 1
                        if count < 3:
                            vr.seek(i_ + count * 1/vr_md['fps'][0])
                        else:
                            log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-1/vr_md['fps'][0]}")
                            vr.seek(i_-1/vr_md['fps'][0], keyframes_only=True)
                            decode_res = next(vr)
                            log.warn(f"Keyframe from {i_} read succesfully at {decode_res['pts']}")
                            res.append(decode_res['data'])
                            break
            imgs = torch.stack(res, axis=0)
            if vr_rotation != 0:
                imgs = torch.rot90(imgs, k=-vr_rotation//90, dims=[2, 3])
            imgs = imgs.permute(0, 2, 3, 1).numpy()[unsort_indexes]
        del vr; gc.collect()

        ls = []
        for i in range(0, 2*self.repeat_sample, 2):
            res = self.aug_fn(imgs[i], imgs[i+1])
            if len(ls) == 0:
                [ls.append([]) for _ in range(len(res))]
            for j, x in enumerate(res):
                ls[j].append(x)
        return tuple(torch.stack(x, dim=0) for x in ls)
 
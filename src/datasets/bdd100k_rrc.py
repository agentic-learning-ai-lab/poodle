import gc
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
def bdd100k_rrc(data_dir, meta_info_file, split,
                rrc_scale,
                rrc_ratio=[1., 1.],
                img_size=None,
                timeofday_filter=None,
                delta_t=[3, 15],
                repeat_sample=None,
                sample_idx=None,
                color_distortion_strength=1.0,
                norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                subset=1,
                backend='decord'):
    assert subset <= 1.0
    assert color_distortion_strength <= 1.0
    assert split == 'train', 'Only support train split for now'
    assert backend in ['decord', 'torchvision', 'torchvision-videoreader']

    # transforms
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    # data augmentations
    s = color_distortion_strength
    h, w = img_size 
    gaussian_kernel = (round_up_odd(0.1*h), round_up_odd(0.1 * w))
    data_aug = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(gaussian_kernel, sigma=(0.1, 2.0))], p=0.5),
        normalize
    ])
    
    rrc_crop = transforms.RandomResizedCrop(size=img_size, scale=rrc_scale, ratio=rrc_ratio, interpolation=InterpolationMode.BILINEAR, antialias=None)
    
    def aug_fn(img1: torch.Tensor, img2: torch.Tensor):
        img1 = tvF.to_tensor(img1)
        img2 = tvF.to_tensor(img2)

        img1, img2 = torch.unbind(
            rrc_crop(torch.stack([img1, img2], dim=0)),
            dim=0
        )
        
        img1_ = data_aug(img1)
        img2_ = data_aug(img2)
        
        return img1_, img2_, img1, img2

    dataset = BDD100KDataset(
        data_dir=data_dir,
        aug_fn=aug_fn,
        mode=split,
        meta_info_file=meta_info_file,
        timeofday_filter=timeofday_filter,
        delta_t=delta_t,
        norm_mean=norm_mean,
        norm_std=norm_std,
        repeat_sample=repeat_sample,
        sample_idx=sample_idx,
        backend=backend,
    )

    if subset != 1.0:
        n = int(len(dataset) * subset)
        data = torchdata.random_split(data, (n, len(data)-n), generator=torch.Generator().manual_seed(1))[0]

    return dataset
class BDD100KDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 aug_fn: Callable,
                 mode: str = "train",
                 meta_info_file: str = None,
                 timeofday_filter: List[str] = None,
                 delta_t: List[int] = [3, 15],
                 norm_mean: List[float] = [0.485, 0.456, 0.406],
                 norm_std: List[float] = [0.229, 0.224, 0.225],
                 repeat_sample=None,
                 sample_idx: List[int] = None,
                 backend='decord',
                 ):
        
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.aug_fn = aug_fn
        self.mode = mode
        self.delta_t = delta_t
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.repeat_sample = repeat_sample
        self.sample_idx = sample_idx
        self.backend = backend
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_dir = os.path.join(data_dir, mode)
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
        if meta_info_file is not None:
            meta_info = np.load(meta_info_file, allow_pickle=True).item()
            self.video_paths = []
            self.video_metadata = []
            for p in video_paths:
                info = meta_info.get(p)
                if info is None:
                    continue
                length = info.get('length', 0)
                if sample_idx:
                    min_start = max(sample_idx)
                else:
                    min_start = 0
                if length <= (min_start + delta_t[1] + 1):
                    continue
                if timeofday_filter is not None and info['attributes'].get('timeofday') not in timeofday_filter:
                    continue
                self.video_paths.append(p)
                self.video_metadata.append(info)
        else:
            self.video_paths = video_paths
            self.video_metadata = [None for _ in video_paths]
        self._dataset_len = len(self.video_paths)
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        if self.backend == 'decord':
            vr = VideoReader(self.video_paths[idx], num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(self.video_paths[idx], "video")
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            if self.video_metadata[idx] is not None and 'rotation' in self.video_metadata[idx]:
                vr_rotation = int(self.video_metadata[idx]['rotation'])
            else:
                try:
                    vr_rotation = -int(ffprobe(self.video_paths[idx]).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
                except:
                    vr_rotation = 0
            vr_md = vr.get_metadata()['video']
            vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1
        if self.repeat_sample is not None:
            if self.sample_idx is not None:
                if len(self.sample_idx) > self.repeat_sample:
                    i_s = np.random.choice(self.sample_idx, size=self.repeat_sample, replace=False)
                elif len(self.sample_idx) == self.repeat_sample:
                    i_s = np.array(self.sample_idx)
                else:
                    raise ValueError(f"len(sample_idx) must be >= repeat_sample")
            else:
                i_s = np.random.randint(0, vr_len-self.delta_t[1], size=self.repeat_sample)
            delta_ts = np.random.randint(self.delta_t[0], self.delta_t[1]+1, size=self.repeat_sample)
            i_s = np.array([index for i, delta_t in zip(i_s, delta_ts) for index in [i, i+delta_t]])
            sort_indexes = np.argsort(i_s).astype(np.int32)
            unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
            if self.backend == 'decord':
                imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
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
                                log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-(count-2)/vr_md['fps'][0]}")
                                vr.seek(i_-(count-2)/vr_md['fps'][0], keyframes_only=True)
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
        else:
            if self.sample_idx is not None:
                i = np.random.choice(self.sample_idx)
            else:
                i = np.random.randint(0, vr_len-self.delta_t[1])
            delta_t = np.random.randint(self.delta_t[0], self.delta_t[1]+1)
            img1 = vr[i].asnumpy()
            img2 = vr[i+delta_t].asnumpy()
            return self.aug_fn(img1, img2)
 